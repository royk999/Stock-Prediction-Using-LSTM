from data_processing import get_df
from data_processing import create_graph_close
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

from multi_lstm import multi_modify_df
from multi_lstm import multi_model_train
from multi_lstm import predict_multi_model
from multi_lstm import return_metrics_multi
from multi_lstm import evaluate_multi_model

from single_improved import single_improved_modify_df
from single_improved import single_improved_model_train
from single_improved import predict_single_improved_model
from single_improved import analyze_single_improved
from single_improved import return_metrics_single_improved
from single_improved import evaluate_single_improved

from datetime import datetime
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def data_analysis(): 
    stock_name = ['AAPL', 'BTC-USD', 'DX-Y.NYB', '^IXIC']
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
    combined_name = ['AAPL', 'AAPL_D', 'bitcoin', 'bitcoin_D', 'USDX', 'USDX_D', 'NASDAQ', 'NASDAQ_D']
    for stock, delta in zip(stock_list, delta_1_list):
        combined_list.append(stock)
        combined_list.append(delta)
    
    create_graph_correlation(stock_name, stock_list, "Closing Price Correlation")
    create_graph_correlation(stock_name, delta_1_list, "Delta 1 Correlation")
   
    #create_graph_correlation(combined_name, combined_list, 'Combined Correlation')

def single_model(model_params, iterations=1):
    stock_name = ['AAPL']
    end = datetime(2023, 10, 30)
    start = datetime(2012, 10, 30)

    df = get_df(stock_name, start, end)

    x_train, y_train, x_val, y_val, x_test, y_test, scalar = single_improved_modify_df(df[0], training_dataset_percentage=0.8, validation_dataset_percentage = 0.1, x_train_len=5)

    model_path = 'model_trained/single_model.keras'

    RMSE = 0
    MAPE = 0
    
    for i in range(iterations):
        print(f'Iteration {i+1} of {iterations}')

        model = load_model(model_path)
        #model = single_model_train(x_train, y_train, x_val, y_val)

        #model.save(model_path)

        predictions, y_test_scaled = predict_single_improved_model(model, x_test, y_test, scalar)

        rmse, mape = return_metrics_single_improved(y_test_scaled, predictions)

        RMSE += rmse
        MAPE += mape

    RMSE /= iterations
    MAPE /= iterations

    path = 'results/results_single_profits.txt'
    evaluate_single_improved(RMSE, MAPE, path = path, **model_params)
    profit_model, profit_random, profit_always, profit_model_rate, profit_random_rate, profit_always_rate, accuracy = analyze_single_improved(y_test_scaled, predictions)

    analyze_singular(y_test_scaled, predictions, 'results/results_Kaggle_model.png')

    print(f'Profit model: {profit_model}, Profit random: {profit_random}, Profit Always: {profit_always}, Profit Model Rate: {profit_model_rate}, Profit Random Rate: {profit_random_rate}, profit_always_rate: {profit_always_rate} Accuracy: {accuracy}', file=open(path, 'a'))

def single_delta_model():
    end = datetime(2022, 10, 30)
    start = datetime(2012, 5, 30)
    stock_name = "AAPL"
    df = get_df_singular(stock_name, start, end)
    x_train, y_train, x_test, y_test, scalar = modify_df_single_delta(df, training_dataset_percentage=0.8, x_train_len=60) # training_dataset_percentage, x_train_len = 60

    model_params = {
        'features_lstm': 10,
        'max_epochs': 40,
        'batch_size': 5,
        'learning_rate': 0.001,
        'clipvalue': 1.0,
        'optimizer' : 'Adam'
    }

    model_path = 'model_trained/single_delta_model.keras'
    model = load_model(model_path)
    model = single_delta_model_train(x_train, y_train, **model_params)
    model.save(model_path) #save the model to a file
    predictions = predict_single_delta(model, x_test, scalar)
    analyze_single_delta(y_test, predictions, scalar, **model_params)

def multi_model(model_params, iterations=1):
    stock_name = ['AAPL', 'DX-Y.NYB', 'MSFT', 'BTC-USD', '^IXIC', 'GOOG', 'AMZN', 'FB', 'TSLA', 'NFLX']
    end = datetime(2023, 10, 30)
    start = datetime(2016, 10, 30)

    stock_list = get_df(stock_name, start, end)
    
    combined_list = []
    
    for stock in stock_list:
        combined_list.append(stock.filter(['Close']).fillna(0))
    
    if np.any(np.isnan(combined_list[1])) or np.any(np.isnan(combined_list[1])):
        print("NaN values found in combined list. Please clean your data.")
        nan_indices = np.where(np.isnan(combined_list[1]))
        print(nan_indices)
        return None
    
    apple_close = stock_list[0].filter(['Close'])

    x_train, y_train, x_val, y_val, x_test, y_test, scalar = multi_modify_df(combined_list, apple_close, training_dataset_percentage=0.8, validation_dataset_percentage = 0.1, x_train_len=5)

    model_path = 'model_trained/multi_model.keras'

    RMSE = 0
    MAPE = 0
    
    for i in range(iterations):
        print(f'Iteration {i+1} of {iterations}')

        #model = load_model(model_path)
        model = multi_model_train(x_train, y_train, x_val, y_val, **model_params)

        model.save(model_path)

        predictions, y_test_scaled = predict_multi_model(model, x_test, y_test, scalar)

        rmse, mape = return_metrics_multi(y_test_scaled, predictions)

        RMSE += rmse
        MAPE += mape

    evaluate_multi_model(RMSE, MAPE, path = 'results/results_multi_model', **model_params)

def single_improved_model(model_params, iterations=1):
    stock_name = ['AAPL']
    end = datetime(2023, 10, 30)
    start = datetime(2012, 10, 30)

    df = get_df(stock_name, start, end)

    x_train, y_train, x_val, y_val, x_test, y_test, scalar = single_improved_modify_df(df[0], training_dataset_percentage=0.8, validation_dataset_percentage = 0.1, x_train_len=5)

    model_path = 'model_trained/single_improved_model.keras'

    RMSE = 0
    MAPE = 0
    
    for i in range(iterations):
        print(f'Iteration {i+1} of {iterations}')

        model = load_model(model_path)
        model = single_improved_model_train(x_train, y_train, x_val, y_val, **model_params)

        model.save(model_path)

        predictions, y_test_scaled = predict_single_improved_model(model, x_test, y_test, scalar)

        rmse, mape = return_metrics_single_improved(y_test_scaled, predictions)

        RMSE += rmse
        MAPE += mape


    RMSE /= iterations
    MAPE /= iterations

    path = 'results/results_single_improved_profits.txt'
    evaluate_single_improved(RMSE, MAPE, path = path, **model_params)
    profit_model, profit_random, profit_always, profit_model_rate, profit_random_rate, profit_always_rate, accuracy = analyze_single_improved(y_test_scaled, predictions)

    print(f'Profit model: {profit_model}, Profit random: {profit_random}, Profit Always: {profit_always}, Profit Model Rate: {profit_model_rate}, Profit Random Rate: {profit_random_rate}, profit_always_rate: {profit_always_rate} Accuracy: {accuracy}', file=open(path, 'a'))

    analyze_singular(y_test_scaled, predictions, 'results/results_optimized_model.png')


def main():
    model_params = {
        'features_lstm': 700,
        'max_epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.1, 
        'clipvalue': 1.0,
        'optimizer' : 'SGD'
    }
    
    '''
    for optimizer in ('Adam', 'Adagrad', 'RMSprop', 'SGD'):
        model_params['optimizer'] = optimizer
        for learning_rate in (0.1, 0.01, 0.001):
            model_params['learning_rate'] = learning_rate
            for batch_size in (1, 4, 8, 16): 
                model_params['batch_size'] = batch_size
                single_improved_model(model_params, 10)

    for features_lstm in (250, ):
        model_params['features_lstm'] = features_lstm
        for batch_size, learning_rate, optimizer in ((1, 0.1, 'SGD'), (4, 0.001, 'Adam'), (8, 0.1, 'SGD')):
            model_params['batch_size'] = batch_size
            model_params['learning_rate'] = learning_rate
            model_params['optimizer'] = optimizer
            single_improved_model(model_params, 10)

    '''

    #data_analysis()
    #single_model(model_params, 1)
    #single_delta_model()
    multi_model(model_params, 1)
    #single_improved_model(model_params, 1)

if __name__ == '__main__':
    main()