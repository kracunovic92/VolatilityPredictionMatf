from sklearn import metrics
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

        
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




