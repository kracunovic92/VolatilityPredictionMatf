import pandas as pd
import numpy as np
import os


data = pd.read_csv("data/GOOG.csv").set_index("Date")

from stock_proccessor import StockDataProcessor

processor = StockDataProcessor()

processor.ma_n(data = data,n = 5)
processor.ma_n(data = data,n = 20)
processor.rsi(data)
processor.bollinger_bands(data = data, n = 5)
processor.bollinger_bands(data= data, n = 20)
processor.average_volume(data = data, n = 5)
processor.calculate_returns(sub_data = data)
processor.calculate_log_returns(data)

print(data)
data.to_csv('data/added_feature.csv',index=True)