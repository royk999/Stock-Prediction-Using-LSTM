from data_processing import get_df
from data_processing import create_graph_close
from data_processing import modify_df
from data_processing import create_graph_delta
from data_processing import create_graph_correlation
from data_processing import delta_df

from single_lstm import get_df_singular
from single_lstm import modify_df_singular
from single_lstm import single_model_train
from single_lstm import predict_singular
from single_lstm import analyze_singular

from single_delta_lstm import modify_df_single_delta
from single_delta_lstm import single_delta_model_train
from single_delta_lstm import predict_single_delta
from single_delta_lstm import analyze_single_delta

from datetime import datetime
from tensorflow.keras.models import load_model
import pandas as pd

def data_analysis(): 
    stock_name = ['AAPL', 'MSFT', 'GOOG', '^IXIC']
    end = datetime(2023, 10, 30)
    start = datetime(2012, 10, 30)

    stock_list = get_df(stock_name, start, end)

    delta_1_list = delta_df(stock_list, stock_name, 1)

    # Save stock_list and delta_1_list as csv files in datasets folder
    #for stock, stock_name in zip(stock_list, stock_name):
    #    stock.to_csv(f'datasets/{stock_name}.csv')

    #create_graph_close(stock_name, stock_list)
    #create_graph_delta(stock_name, stock_list)

    combined_list = []
    combined_name = ['AAPL', 'AAPL_D', 'MSFT', 'MSFT_D', 'GOOG', 'GOOG_D', 'NASDAQ', 'NASDAQ_D']
    for stock, delta in zip(stock_list, delta_1_list):
        combined_list.append(stock)
        combined_list.append(delta)
    
    create_graph_correlation(stock_name, stock_list, "Closing Price Correlation")
    create_graph_correlation(stock_name, delta_1_list, "Delta 1 Correlation")
    create_graph_correlation(combined_name, combined_list, 'Combined Correlation')

def single_model():
    end = datetime(2022, 10, 30)
    start = datetime(2012, 5, 30)
    stock_name = "AAPL"
    df = get_df_singular(stock_name, start, end)
    x_train, y_train, x_test, y_test, scalar = modify_df_singular(df, 0.8, 60) # training_dataset_percentage = 0.8, x_train_len = 60
    
    #model = single_model_train(x_train, y_train)
    #model.save('single_model.keras') # save the model to a file

    model_path = 'model_trained/single_model.keras'
    model = load_model(model_path)

    predictions = predict_singular(model, x_test, scalar)

    analyze_singular(y_test, predictions)

def single_delta_model():
    end = datetime(2022, 10, 30)
    start = datetime(2012, 5, 30)
    stock_name = "AAPL"
    df = get_df_singular(stock_name, start, end)
    x_train, y_train, x_test, y_test, scalar = modify_df_single_delta(df, training_dataset_percentage=0.8, x_train_len=60) # training_dataset_percentage, x_train_len = 60
    model = single_delta_model_train(x_train, y_train)
    model.save('model_trained/single_delta_model.keras') # save the model to a file

    #model_path = 'model_trained/single_delta_model.keras'
    #model = load_model(model_path)

    predictions = predict_single_delta(model, x_test, scalar)
    
    analyze_single_delta(y_test, predictions)

def multi_model():
    stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    stock_name = ['APPLE', 'MICROSOFT', 'GOOGLE', 'AMAZON']
    end = datetime(2023, 10, 30)
    start = datetime(2022, 10, 30)

    company_list = get_df(stock_list, stock_name, start, end)

    df = modify_df(company_list, 0.8)

def main():
    #data_analysis()
    #single_model()
    single_delta_model()
    #multi_model()

if __name__ == '__main__':
    main()