from sklearn import metrics
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


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
        self.period_test = period_test

        # Predefined generated output
        self.training_set = None  # data frames
        self.testing_set = None  # data frames
        self.train_matrix = None
        self.train_y = None
        self.test_matrix = None
        self.test_y = None
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

    
    def generate_dataset(self):

        self.jump_detection()  # outliers

        self.future_averages() # targets

        if self.log_transform: # here is the problemos (maybe)
            self.future_values["Target"] = np.log(self.future_values["Target"])
            s_targets = MinMaxScaler()
            self.applied_scaler_targets = s_targets
            self.future_values["Target"] = s_targets.fit_transform(
                self.future_values["Target"].values.reshape(-1, 1)
            )

        self.data_scaling()  # data scaling after future value generation
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

        data_train = data.loc[(data.index >= self.period_train[0]) &(data.index <= self.period_train[1])].reset_index(drop= True)
        data_test = data.loc[(data.index >= self.period_test[0]) &(data.index <= self.period_test[1])].reset_index(drop = True)

        self.training_set = data_train
        self.testing_set = data_test


    def generate_validation_split(self):

        self.train_matrix = self.training_set.drop(columns={"Target"}).values
        self.train_y = self.training_set[["Target"]].values

        self.test_matrix = self.testing_set.drop(columns={"Target"}).values
        self.test_y = self.testing_set[["Target"]].values

        n_features = 1

        train_shape_rows = self.train_matrix.shape[0]
        train_shape_columns = self.train_matrix.shape[1]

        self.train_matrix = self.train_matrix.reshape(
            (train_shape_rows, train_shape_columns, n_features)
        )

        test_shape_rows = self.test_matrix.shape[0]
        test_shape_columns = self.train_matrix.shape[1]

        self.test_matrix = self.test_matrix.reshape(
            (test_shape_rows, test_shape_columns, n_features)
        )


    def prepare_all(self):

        self.data_transformation()

        if self.training_set is None:
            self.generate_training_test_split()

        if self.train_matrix is None:
            self.generate_validation_split()

        
class LSTM:
    def __init__(
        self,
        #training_set,
        #testing_set,
        train_matrix,
        train_y,
        test_matrix,
        test_y,
        activation=tf.nn.elu,
        epochs=50,
        learning_rate=0.01,
        layer_one=40,
        layer_two=40,
        layer_three=0,
        layer_four=0,
    ):
        #self.training_set = training_set
        #self.testing_set = testing_set
        self.train_matrix = train_matrix
        self.train_y = train_y
        self.test_matrix = test_matrix
        self.test_y = test_y
        self.activation = activation
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layer_one = int(layer_one)
        self.layer_two = int(layer_two)
        self.layer_three = int(layer_three)
        self.layer_four = int(layer_four)

        # Predefined output
        self.fitted_model = None
        self.prediction_train = None
        self.prediction_test = None
        self.test_accuracy = None
        self.train_accuracy = None
        self.fitness = None

    def train_lstm(self):

        m = tf.keras.models.Sequential()
        m.add(
            tf.keras.layers.LSTM(
                self.layer_one,
                activation=self.activation,
                return_sequences=True,
                input_shape=(int(self.train_matrix.shape[int(1)]), int(1)),
            )
        )
        if self.layer_two > 0:
            if self.layer_three > 0:
                if self.layer_four > 0:
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_two,
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_three,
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_four, activation=self.activation,
                        )
                    )
                else:
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_two,
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_three, activation=self.activation,
                        )
                    )
            else:
                m.add(tf.keras.layers.LSTM(self.layer_two, activation=self.activation))
        m.add(tf.keras.layers.Dense(1, activation="linear"))

        o = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                decay=0.0,
                amsgrad=False,
            )
        
        m.compile(optimizer=o, loss=tf.keras.losses.logcosh)

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10, verbose=1,
        )

        m.fit(
            self.train_matrix,
            self.train_y,
            epochs=self.epochs,
            verbose=1,
            callbacks=[es],  # added
            validation_data=(self.test_matrix, self.test_y,),
        )

        self.fitted_model = m


    def predict_lstm(self):

        if self.fitted_model is None:
            self.train_lstm()

        self.prediction_train = self.fitted_model.predict(self.train_matrix)
        self.prediction_test = self.fitted_model.predict(self.test_matrix)

    
    def make_accuracy_measures(self):
        if self.prediction_test is None:
            self.predict_lstm()
        
        test_accuracy = {
            "MSE": metrics.mean_squared_error(
                self.test_y, self.prediction_test
            ),
            "MAE": metrics.mean_absolute_error(
                self.test_y, self.prediction_test
            ),
            "RSquared": metrics.r2_score(
                self.test_y, self.prediction_test
            ),
        }
        train_accuracy = {
            "MSE": metrics.mean_squared_error(
                self.train_y, self.prediction_train
            ),
            "MAE": metrics.mean_absolute_error(
                self.train_y, self.prediction_train
            ),
            "RSquared": metrics.r2_score(
                self.train_y, self.prediction_train
            ),
        }

        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy
        self.fitness = self.train_accuracy["RSquared"] + self.test_accuracy["RSquared"]




