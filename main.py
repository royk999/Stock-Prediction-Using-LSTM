from data_processing import get_df
from data_processing import create_graph_close
from data_processing import modify_df
from datetime import datetime

def main():
    stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    company_name = ['APPLE', 'MICROSOFT', 'GOOGLE', 'AMAZON']
    end = datetime(2023, 10, 30)
    start = datetime(2023, 9, 30)

    df, company_list = get_df(stock_list, company_name, start, end)

    create_graph_close(company_name, company_list)

    new_data, training_data_len = modify_df(df, 0.8) # training_dataset_percentage = 0.8


if __name__ == '__main__':
    main()