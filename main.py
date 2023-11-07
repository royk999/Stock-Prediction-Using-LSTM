from data_processing import get_df
from data_processing import create_graph_close
from data_processing import modify_df
from data_processing import create_graph_comp

from single_lstm import get_df_singular
from single_lstm import modify_df_singular
from single_lstm import single_lstm_model

from datetime import datetime


def singular_model():
    df = get_df_singular()
    x_train, y_train, x_test, y_test = modify_df_singular(df, 0.8, 60) # training_dataset_percentage = 0.8, x_train_len = 60
    print(x_test.shape)
    #model = single_lstm_model(x_train, y_train, x_test, y_test)



def data_analysis(): 
    stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    company_name = ['APPLE', 'MICROSOFT', 'GOOGLE', 'AMAZON']
    end = datetime(2023, 10, 30)
    start = datetime(2023, 5, 30)

    df, company_list = get_df(stock_list, company_name, start, end)

    create_graph_close(company_name, company_list)

    create_graph_comp()

    
def multi_model():
    data_analysis()

def main():
    #singular_model()
    multi_model()

if __name__ == '__main__':
    main()