from data_processing import get_data
from data_processing import create_graph_high
from datetime import datetime

def main():
    stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    company_name = ['APPLE', 'MICROSOFT', 'GOOGLE', 'AMAZON']
    end = datetime(2023, 10, 30)
    start = datetime(2023, 9, 30)

    data, company_list = get_data(stock_list, company_name, start, end)

    print(data.head())
    data.info()

    create_graph_high(company_name, company_list)
    

if __name__ == '__main__':
    main()