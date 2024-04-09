
import numpy as np
import pandas as pd


from sklearn.model_selection import TimeSeriesSplit



data = pd.read_csv('data/GOOG.csv').set_index('Date')
window_size = 100
total_data_points = len(data)


for i in range(total_data_points - window_size):
    train_set = data[i:i+window_size]
    validation_set = data[+ window_size:i+window_size+1]

   