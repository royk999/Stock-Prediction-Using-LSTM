from data_processing import get_df
from data_processing import create_graph_close
from data_processing import modify_df
from data_processing import create_graph_delta
from data_processing import create_graph_correlation
from data_processing import delta_df

from single_lstm import get_df_singular
from single_lstm import modify_df_singular
from single_lstm import single_lstm_model
from single_lstm import predict_singular
from single_lstm import analyze_singular

from datetime import datetime
from tensorflow.keras.models import load_model

def singular_model():
    end = datetime(2022, 10, 30)
    start = datetime(2012, 5, 30)
    df = get_df_singular(start, end)
    x_train, y_train, x_test, y_test, scalar = modify_df_singular(df, 0.8, 60) # training_dataset_percentage = 0.8, x_train_len = 60
    
    #model = single_lstm_model(x_train, y_train)
    
    model_path = 'model_trained/singular_model.keras'
    model = load_model(model_path)

    model.save('singular_model.keras') # save the model to a file

    predictions = predict_singular(model, x_test, scalar)

    analyze_singular(y_test, predictions)


def data_analysis(): 
    stock_name = ['AAPL', 'MSFT', 'GOOG', 'AMZN', '^IXIC', '^GSPC', '^DJI']
    end = datetime(2023, 10, 30)
    start = datetime(2012, 10, 30)

    stock_list = get_df(stock_name, start, end)

    delta_1_list = delta_df(stock_list, stock_name, 1)

    # Save stock_list and delta_1_list as csv files in datasets folder
    for stock, stock_name in zip(stock_list, stock_name):
        stock.to_csv(f'datasets/{stock_name}.csv')


    #create_graph_close(stock_name, stock_list)
    #create_graph_delta(stock_name, stock_list)
    create_graph_correlation(stock_name, stock_list, "Closing Price Correlation")
    create_graph_correlation(stock_name, delta_1_list, "Delta 1 Correlation")

    
def multi_model():
    stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    stock_name = ['APPLE', 'MICROSOFT', 'GOOGLE', 'AMAZON']
    end = datetime(2023, 10, 30)
    start = datetime(2022, 10, 30)

    company_list = get_df(stock_list, stock_name, start, end)

    df = modify_df(company_list, 0.8)

def main():
    data_analysis()
    #singular_model()
    #multi_model()

if __name__ == '__main__':
    main()