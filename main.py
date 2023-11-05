from data_processing import get_data
from datetime import datetime

def main():
    stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    end = datetime(2023, 10, 30)
    start = datetime(2023, 9, 30)

    data = get_data(stock_list, start, end)

    print(data.head())
    data.info() 

if __name__ == '__main__':
    main()