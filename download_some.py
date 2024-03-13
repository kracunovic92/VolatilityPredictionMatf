
from datetime import datetime
from data import download_stock_data

company_list = ["AAPL", "GOOG", "MSFT", "AMZN"]

end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)


for company in company_list:
    download_stock_data(company,start_date= start, end_date= end,output_folder= "data")


