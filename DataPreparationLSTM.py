from sklearn import metrics
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit


class DataPreparationLSTM:
    def __init__(
        self,
        df: pd.DataFrame,
        future = 1,
        lag = 20,
        semi_variance: bool = False,
        jump_detect: bool = True,
        log_transform: bool = True,
        min_max_scaler: bool = True,
        period_train=list(
            [
                pd.to_datetime("20030910", format="%Y%m%d"),
                pd.to_datetime("20080208", format="%Y%m%d"),
            ]
        ),
        period_validation = list(
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
        ),
    ):
        self.df = df
        self.future = future
        self.lag = lag
        self.semi_variance = semi_variance
        self.jump_detect = jump_detect
        self.log_transform = log_transform
        self.min_max_scaler = min_max_scaler
        self.period_train = period_train
        self.period_validation = period_validation
        self.period_test = period_test

        # Predefined generated output
        self.training_set = None  # data frames
        self.testing_set = None  # data frames
        self.validation_set = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.splits = None
        self.future_values = None
        self.historical_values = None
        self.df_processed_data = None,
        self.applied_scaler_features = None
        self.applied_scaler_targets = None

    def jump_detection(self):
        
        df_tmp = self.df.copy()
        df_tmp["threshold"] = df_tmp["RV"].rolling(window=200).std() * 4
        df_tmp.threshold = np.where(df_tmp.threshold.isna(), 1, df_tmp.threshold)
        df_tmp["larger"] = np.where(df_tmp["RV"] > df_tmp.threshold, True, False)
        df_tmp = df_tmp[df_tmp.larger == False]

        df_tmp.drop(columns={"threshold", "larger"}, axis=1, inplace=True)

        self.df = df_tmp.copy()

    def data_transformation(self):
        
        df_temp = self.df.copy()
        df_temp['Returns'] = np.log(df_temp['close']) - np.log(df_temp['close'].shift(1))

        # Remove returns in between days
        df_temp.loc[df_temp['date'] != df_temp['date'].shift(1),'Returns'] = None

        df_temp['returns**2']= df_temp['Returns'] **2
        Rv = df_temp.groupby('date')['returns**2'].sum().rename('RV')
        df_temp['sum'] = df_temp['Returns'].abs() * df_temp['Returns'].abs().shift(1)
        coef = np.sqrt(2/np.pi) ** (-2)
        Bv = coef * df_temp.groupby('date')['sum'].sum().rename('BV')
        df_temp['positive_returns'] = (df_temp['returns**2']) * (df_temp['Returns'] > 0)
        df_temp['negative_returns'] = (df_temp['returns**2']) * (df_temp['Returns'] < 0)
        Rs_p = df_temp.groupby('date')['positive_returns'].sum().rename('RVp')
        Rs_n = df_temp.groupby('date')['negative_returns'].sum().rename('RVn')

        data= pd.DataFrame(Rv)
        data = data.join(Rs_p, on = 'date')
        data = data.join(Rs_n, on = 'date')
        data = data.join(Bv, on = 'date')

        self.df = data

    '''
    #Wrong
    def data_scaling(self):

        if self.log_transform:
            self.df['RV'] = np.log(self.df['RV'])
            if self.semi_variance:
                self.df['RVp'] = np.log(self.df['RVp'])
                self.df['RVn'] = np.log(self.df['RVn'])

        if self.min_max_scaler:
            s = MinMaxScaler()
            self.applied_scaler_features = s
            self.df['RV'] = s.fit_transform(self.df['RV'].values.reshape(-1, 1))
            if self.semi_variance:
                self.df['RVp'] = s.fit_transform(
                    self.df['RVp'].values.reshape(-1, 1)
                )
                self.df['RVn'] = s.fit_transform(
                    self.df['RVn'].values.reshape(-1, 1)
                )

    '''
                
    def future_averages(self):

        data = self.df[["RV"]].copy()

        for i in range(self.future):
            data["Target{}".format(i + 1)] = data['RV'].shift(-(i + 1))
        data = data.dropna()

        help_df = data.drop(["RV"], axis=1)

        df_output = data[["RV"]]
        df_output["Target"] = help_df.mean(axis=1)

        df_output = df_output.drop(["RV"], axis=1)

        self.future_values = df_output

    def historical_lag_transformation(self):

        df = self.df[["RV"]].copy()
        for i in range((self.lag - 1)):
            df["lag_{}".format(i + 1)] = df['RV'].shift(+(i + 1))

        self.historical_values = df

    def back_transformation(self, data):
        if self.log_transform:
            if self.min_max_scaler:
                return np.exp(self.applied_scaler_targets.inverse_transform(data))
        else:
            if self.min_max_scaler:
                return self.applied_scaler_targets.inverse_transform(data)
    
    def generate_dataset(self):

        self.jump_detection()  # outliers
        self.future_averages() # targets
        '''
        if self.log_transform: # here is the problemos (maybe)
            self.future_values["Target"] = np.log(self.future_values["Target"])
            s_targets = MinMaxScaler()
            self.applied_scaler_targets = s_targets
            self.future_values["Target"] = s_targets.fit_transform(
                self.future_values["Target"].values.reshape(-1, 1)
            )
        '''
        
        if self.log_transform:
            self.future_values['Target'] = np.log(self.future_values['Target'])
            self.df['RV'] = np.log(self.df['RV'])
            if self.semi_variance:
                self.df['RVp'] = np.log(self.df['RVp'])
                self.df['RVn'] = np.log(self.df['RVn'])
        
        # self.data_scaling()  # data scaling after future value generation
        self.historical_lag_transformation()

        # merging the two data sets
        data_set_complete = self.future_values.merge(
            self.historical_values, how="right", on="date"
        )
        data_set_complete = data_set_complete.dropna()
        # data_set_complete.reset_index(drop=True, inplace=True)

        if self.semi_variance:
            df_tmp = self.df[["RVn"]]
            data_set_complete = data_set_complete.merge(df_tmp, on='date')

        self.df_processed_data = data_set_complete

    def generate_training_test_split(self):

        self.generate_dataset()

        data = self.df_processed_data.copy()
        data.index = pd.to_datetime(data.index, format = "%Y-%m-%d")

        data_train = data.loc[(data.index >= self.period_train[0]) &(data.index <= self.period_train[1])]
        data_validation = data.loc[(data.index >= self.period_validation[0])& (data.index <= self.period_validation[1])]
        data_test = data.loc[(data.index >= self.period_test[0]) &(data.index <= self.period_test[1])]
        

        self.validation_set = data_validation
        self.training_set = data_train
        self.testing_set = data_test


    def generate_validation_split(self):

        x_train = self.training_set.drop(columns={"Target"}).values
        y_train = self.training_set[["Target"]].values

        x_val = self.validation_set.drop(columns = {"Target"}).values
        y_val = self.validation_set[['Target']].values

        x_test = self.testing_set.drop(columns={"Target"}).values
        y_test = self.testing_set[["Target"]].values

        if self.min_max_scaler:
            sc_features = MinMaxScaler()
            sc_target = MinMaxScaler()
            sc_features.fit(x_train)
            sc_target.fit(y_train)
            self.applied_scaler_features = sc_features
            self.applied_scaler_targets = sc_target
            self.x_train = sc_features.transform(x_train)
            self.x_val = sc_features.transform(x_val)
            self.x_test = sc_features.transform(x_test)
            
            self.y_train = sc_target.transform(y_train)
            self.y_val = sc_target.transform(y_val)
            self.y_test = sc_target.transform(y_test)

            

        n_features = 1

        train_shape_rows = self.x_train.shape[0]
        train_shape_columns = self.x_train.shape[1]

        self.x_train = self.x_train.reshape(
            (train_shape_rows, train_shape_columns, n_features)
        )
        validation_shape_rows = self.x_val.shape[0]
        validation_shape_columns = self.x_val.shape[1]

        self.x_val = self.x_val.reshape(
            (validation_shape_rows, validation_shape_columns, n_features)
        )

        test_shape_rows = self.x_test.shape[0]
        test_shape_columns = self.x_train.shape[1]

        self.x_test = self.x_test.reshape(
            (test_shape_rows, test_shape_columns, n_features)
        )

    def generate_splits(self):
        
        tscv = TimeSeriesSplit(n_splits=5)

        folds = []

        fold_indices = []

        validation_x = self.x_val.copy()
        validation_y = self.y_val.copy()

        train_x = self.x_train.copy()
        train_y = self.y_train.copy()

        for train_idx, test_idx in tscv.split(validation_x):
            fold_indices.append((train_idx, test_idx))


        for fold, (train_idx, test_idx) in enumerate(fold_indices):
            # Combine the training data and previous validation folds for training
            if fold > 0:
                train_combined_x = np.concatenate([train_x, validation_x[:test_idx[0]]])
                train_combined_y = np.concatenate([train_y, validation_y[:test_idx[0]]])
            else:
                train_combined_x = train_x  # Use only train_data for the first fold
                train_combined_y = train_y  # Use only train_data for the first fold

            # Use validation_data[test_idx] as the testing data for this fold
            test_data_x = validation_x[test_idx]
            test_data_y = validation_y[test_idx]

            folds.append([train_combined_x, train_combined_y, test_data_x, test_data_y])


        self.splits = folds

        



    def prepare_all(self):

        self.data_transformation()

        if self.training_set is None:
            self.generate_training_test_split()

        if self.x_train is None:
            self.generate_validation_split()

        if self.splits is None:
            self.generate_splits()
