import pandas as pd
import numpy as np
import os

import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()
from datetime import datetime


def download_stock_data(symbol, start_date, end_date, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = yf.download(symbol, start= start_date, end= end_date)

    output_file = os.path.join(output_folder, f"{symbol}.csv")
    data.to_csv(output_file)

    print(f"Stock data  downloaded {output_file}")

tech_list = ['AAPL','GOOG','MSFT','AMZN']


def make_csv_all(input_foler,stock_list,output_file):

    combined_data = pd.DataFrame()
    
    for stock in stock_list:

        file = os.path.join(input_foler,f"{stock}.csv")
        data =pd.read_csv(file)
        data['Company'] = stock
        combined_data = pd.concat([combined_data,data], ignore_index=True)

    combined_data.to_csv(output_file,index= False)
    return combined_data



def create_constituents_dict(stock_data, start_year, end_year):

    con_dict = {}

    for test_year in range(start_year, end_year +1):
        test_year_data = stock_data[stock_data['Date'].str.startswith(str(test_year))]

        stock_symbols = set