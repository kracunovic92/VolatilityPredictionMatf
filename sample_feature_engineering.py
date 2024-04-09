import pandas as pd
import numpy as np
import warnings
from datetime import datetime


class DataProcessor:
    data = None
    coef = np.sqrt(2/np.pi) ** (-2)
    thresh_stat = 0
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data.drop(['Unnamed: 0','money','open','high','low'],axis = 1, inplace=True)



    def getData(self):
        return self.data
    
    def define_time_window(self,lower_bound, upper_bound):
        self.data = self.data[(self.data['time'].dt.time >= pd.to_datetime(lower_bound).time())
                              & (self.data['time'].dt.time <= pd.to_datetime(upper_bound).time())
                              ]

    def calculate_returns(self):
        self.data['returns'] = np.log(self.data['close']- np.log(self.data['close'].shift(1)))
        self.data.loc[self.data['date'] != self.data['date'].shift(1),'returns'] = None
        self.data['returns**2'] = self.data['returns']**2
        self.data['sum_for_BV'] = self.data['returns'].abs() *self.data['returns'].abs().shift(1)
        self.data['positive_returns'] = self.data['returns**2'] * (self.data['returns'] > 0)
        self.data['negative_returns'] = self.data['returns**2'] * (self.data['returns'] < 0)


    def calculate_daily_data(self):
        rv = self.data.groupby('date')['returns**2'].sum().rename('RV')
        bv = self.coef* self.data.groupby('date')['sum_for_BV'].sum().rename('BV')
        rs_p = self.data.groupby('date')['positive_returns'].sum().rename('RV+')
        rs_n = self.data.groupby('date')['negative_returns'].sum().rename('RV-')

        rv = pd.DataFrame(rv)
        rv = rv.join(rs_p, on = 'date')
        rv = rv.join(rs_n, on = 'date')
        rv = rv.join(bv, on = 'date')
        
        return rv
    
    def set_threshold(self, data, n = 4, window = 200):
        rolling_std = data['RV'].rolling(window = window).sum()
        data_filter = data[data['RV'] <= n * rolling_std]
        self.thresh_stat = (data.shape[0]- data_filter.shape[0]) / data.shape[0] * 100
        return data_filter

    def calculate_averages(self,data,type,n):

        types = ['RV','RV+','RV-','BV']
        days = [5,22]
        for day,type in days, types:
            data[f'{type}_{day}'] = data[f'{type}'].rolling(n=day).mean()
        return data
