from tqdm import tqdm
import pandas as pd
import os
import numpy as np

class StockDataProcessor:

    def ma_n(self,data,n):
        data[f'MA_{n}'] = data['Close'].rolling(window= n).mean()

    def rsi(self, data):
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0 ,0)

        average_gain = gain.rolling(window = 14).mean()
        average_lost = loss.rolling(window = 14).mean()
        rs = average_gain / average_lost
        data['RSI'] = 100 - ( 100 / (1 + rs))

    
    def macd(self, data, window_short, window_long):
        short_ema = data.ewm(span= window_short, adjust= False).mean()
        long_ema = data.ewm(span = window_long, adjust = False).mean()

        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span = 9, adjust= False).mean()

        macd = macd_line - signal_line
        return macd
    
    def bollinger_bands(self, data, n):
        # Check if 'MA' column exists, if not, calculate it
        if f'MA_{n}' not in data.columns:
            data[f'MA_{n}'] = data['Close'].rolling(window=n).mean()

        data['Upper_band'] = data[f'MA_{n}'] + 2 * data['Close'].rolling(window=n).std()
        data['Lower_band'] = data[f'MA_{n}'] - 2 * data['Close'].rolling(window=n).std()

    def average_volume(self, data, n):
        data[f'Average_Volume_{n}'] = data['Volume'].rolling(window=n).mean()

    def extract_time_info(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data['day'] = data['date'].dt.dayofweek
        data['Month'] = data['date'].dt.month
        data['Year'] = data['date'].dt.year

    # Function for calculating returns
    def calculate_returns(self, sub_data):
        sub_data['Price_Return'] = sub_data['Adj Close'].pct_change()
        sub_data['Volume_Return'] = sub_data['Volume'].pct_change()
        return sub_data
    
    def calculate_log_returns(self, data):
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    # Function for generating 1/0 target
    def generate_target(sub_data):
        return_median = sub_data['Price_Return'].median()
        sub_data['Target'] = sub_data['Price_Return'].apply(lambda x: 1 if x >= return_median else 0)
        return sub_data

    # Function for normalizing data
    def normalize(self,a , mean, std):
        return (a - mean) / std