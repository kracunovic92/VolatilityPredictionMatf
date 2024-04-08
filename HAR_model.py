import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn import metrics


class HARModel:

    def __init__(
        self,
        raw_data, # 5-min data sample without any changes
        future= 1,
        lags=[4, 20,],
        feature="RV",
        semi_variance=False,
        jump_detect=True,
        log_transformation=False,
        time_windows= [
            pd.to_datetime('09:30').time(),
            pd.to_datetime('16:00').time()
        ],
        period_train=list(
            [
                pd.to_datetime("20030910", format="%Y%m%d"),
                pd.to_datetime("20080208", format="%Y%m%d"),
            ]
        ),
        period_test=list(
            [
                pd.to_datetime("20080209", format="%Y%m%d"),
                pd.to_datetime("20101231", format="%Y%m%d"),
            ]
        )
    ):
        self.raw_data = raw_data
        self.future = future
        self.lags = lags
        self.feature = feature
        self.semi_variance = semi_variance
        self.jump_detect = jump_detect
        self.log_transformation = log_transformation
        self.period_train = period_train
        self.period_test = period_test
        self.time_window = time_windows
        self.training_set = None 
        self.testing_set = None  
        self.prediction_train = (
            None  
        )
        self.prediction_test = None
        self.model = None  # stats model instance
        self.estimation_results = None  # table
        self.test_accuracy = None  # dictionary
        self.train_accuracy = None
        self.output_dataset = None  # DataFrame (data frame which contains all the features needed for the regression)
        self.data = None
        self.data_filltered_on_jump = None

    def data_transfomation(self):
        
        df_temp = self.raw_data.copy()
        df_temp['Returns'] = np.log(df_temp['close']) - np.log(df_temp['close'].shift(1))

        #we do not want to calculate return in between days because we just care about
        # intraday volatility
        df_temp.loc[df_temp['date'] != df_temp['date'].shift(1),'Returns'] = None

        df_temp['returns**2']= df_temp['Returns'] **2

        Rv = df_temp.groupby('date')['returns**2'].sum().rename('RV')
        df_temp['sum'] = df_temp['Returns'].abs() * df_temp['Returns'].abs().shift(1)
        coef = np.sqrt(2/np.pi) ** (-2)
        Bv = coef * df_temp.groupby('date')['sum'].sum().rename('BV')
        df_temp['positive_returns'] = (df_temp['returns**2']) * (df_temp['Returns'] > 0)
        df_temp['negative_returns'] = (df_temp['returns**2']) * (df_temp['Returns'] < 0)
        Rs_p = df_temp.groupby('date')['positive_returns'].sum().rename('RV+')
        Rs_n = df_temp.groupby('date')['negative_returns'].sum().rename('RV-')

        #logatirhmic range estimators are an important class of estimators that require less data than RV
        # LR(t) = 1/(4 log2) (H(t) -L(t))** 2 // A practical guide to harnessing the HAR RV- Adam Clements

        data= pd.DataFrame(Rv)
        data = data.join(Rs_p,on = 'date')
        data = data.join(Rs_n,on = 'date')
        data = data.join(Bv, on = 'date')

        self.data = data

    def lag_average(self, log_transform =  False):

        tmp = self.data[[self.feature]]

        if log_transform:
            tmp[self.feature] = np.log(tmp[self.feature])
        
        tmp['RV_t']= tmp[self.feature].shift(1)
        tmp['RV_w']= tmp[self.feature].rolling(window=self.lags[0]).mean()

        rm = tmp[self.feature].rolling(window=self.lags[1]).sum()
        rw = tmp[self.feature].rolling(window= self.lags[0]).sum()

        tmp['RV_m']= (rm - rw) /  (self.lags[1]- self.lags[0])

        # tmp.drop([self.feature], axis=1, inplace=True)

        return  tmp

    def jump_detection(self):
        tmp = self.data.copy()
        threshold =tmp['RV'].rolling(window = 200).std() * 4
        threshold.fillna(1,inplace=True)
        on_start_rows = tmp.shape[0]
        tmp['larger'] =tmp['RV'] > threshold

        tmp = tmp[tmp['larger'] == False]
        tmp.drop(['larger'], axis=1, inplace=True)

        on_end_rows = tmp.shape[0]
        self.data_filltered_on_jump =  (on_start_rows- on_end_rows) / on_start_rows * 100
        self.data = tmp.copy()

    def generate_dataset(self):
        data = self.lag_average(log_transform=False)


        df_help = pd.DataFrame()

        for x in range(self.future):
            df_help[str(x)] = data.RV_t.shift(-(1 + x))
        df_help = df_help.dropna()

        self.output_dataset = self.lag_average(log_transform=self.log_transformation)


        if self.log_transformation:
            self.output_dataset["Target"] = np.log(df_help.mean(axis=1))
            self.output_dataset = self.output_dataset.dropna()
        else:
            self.output_dataset["Target"] = df_help.mean(axis=1)
            self.output_dataset = self.output_dataset.dropna()

    def generate_training_test_split(self):
        self.generate_dataset()

        data = self.output_dataset.copy()
        data.index = pd.to_datetime(data.index, format = "%Y-%m-%d")

        training_set = data.loc[(data.index >= self.period_train[0]) &(data.index <= self.period_train[1])].reset_index(drop= True)
        test_set = data.loc[(data.index >= self.period_test[0]) &(data.index <= self.period_test[1])].reset_index(drop = True)

        self.training_set = training_set
        self.testing_set = test_set
        
    def estimate_model(self):

        self.generate_training_test_split()

        if self.semi_variance:
            result = smf.ols(
                formula="Target ~ RV+ + RV- + RV_w + RV_m",
                data = self.training_set
            ).fit()
        else:
            result = smf.ols(
                formula="Target ~ RV_t + RV_w + RV_m",
                data = self.training_set
            ).fit()
        self.model = result

        results_robust = result.get_robustcov_results(
            cov_type= "HAC", maxlags = 2 * (self.future-1)
        )
        self.estimation_results = results_robust.summary().as_latex()


    def predict_values(self):
        self.estimate_model()
        if self.log_transformation:
            if self.semi_variance:
                self.prediction_train = np.exp(
                    self.model.predict(
                        self.training_set[["RV+", "RV-", "RV_w", "RV_m"]]
                    )
                )
                self.prediction_test = np.exp(
                    self.model.predict(
                        self.testing_set[["RV+", "RV-", "RV_w", "RV_m"]]
                    )
                )
            else:
                self.prediction_train = np.exp(
                    self.model.predict(self.training_set[["RV_t", "RV_w", "RV_m"]])
                )
                self.prediction_test = np.exp(
                    self.model.predict(self.testing_set[["RV_t", "RV_w", "RV_m"]])
                )
        else:
            if self.semi_variance:
                self.prediction_train = self.model.predict(
                    self.training_set[["RV+", "RV-", "RV_w", "RV_m"]]
                )
                self.prediction_test = self.model.predict(
                    self.testing_set[["RV+", "RV-", "RV_w", "RV_m"]]
                )
            else:
                self.prediction_train = self.model.predict(
                    self.training_set[["RV_t", "RV_w", "RV_m"]]
                )
                self.prediction_test = self.model.predict(
                    self.testing_set[["RV_t", "RV_w", "RV_m"]]
                )
    def make_accurate_measures(self):   

        if self.log_transformation:
            self.testing_set["Target"] = np.exp(self.testing_set["Target"])
            self.training_set["Target"] = np.exp(self.testing_set["Target"])

        test_accuracy = {
            "MSE": metrics.mean_squared_error(
                self.testing_set["Target"], self.prediction_test
            ),
            "MAE": metrics.mean_absolute_error(
                self.testing_set["Target"], self.prediction_test
            ),
            "RSquared": metrics.r2_score(
                self.testing_set["Target"], self.prediction_test
            ),
        }
        train_accuracy = {
            "MSE": metrics.mean_squared_error(
                self.training_set["Target"], self.prediction_train
            ),
            "MAE": metrics.mean_absolute_error(
                self.training_set["Target"], self.prediction_train
            ),
            "RSquared": metrics.r2_score(
                self.training_set["Target"], self.prediction_train
            ),
        }

        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy
