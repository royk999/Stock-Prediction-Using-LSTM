from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler # for data scaling
import yfinance as yf

def get_df(stock_name, start, end):
    # Get data from Yahoo Finance
    yf.pdr_override()

    data = {}

    for stock in stock_name:
        data[stock] = yf.download(stock, start, end)

    stock_list = [data[stock] for stock in stock_name]

    for stock, stock_name in zip(stock_list, stock_name):
        stock["stock_name"] = stock_name
        
    return stock_list

def delta_df(stock_list, stock_name, delta_num):
    delta_list = []

    for stock in stock_list:
        delta_list.append(pd.DataFrame(stock['Close'].diff(delta_num)))
        #set all nan values to zeroes
        delta_list[-1] = delta_list[-1].fillna(0)
        
    for delta, stock_name in zip(delta_list, stock_name):
        delta["stock_name"] = stock_name

    return delta_list
    
def modify_df(stock_list, output_data, training_dataset_percentage, x_train_len):   
    len_data = len(stock_list[0])

    data = pd.concat(stock_list, axis=1)
    dataset = data.values
    output_dataset = output_data.values

    training_data_len = int(np.ceil(len_data * training_dataset_percentage))

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    print(output_dataset)
    
    training_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(x_train_len, training_data_len):
        x_train.append(training_data[i - x_train_len:i, :])
        y_train.append(output_dataset[i])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Create the testing data set
    test_data = scaled_data[training_data_len - x_train_len: , :]
    x_test = []
    y_test = []
    for i in range(x_train_len, len(test_data)):
        x_test.append(test_data[i-x_train_len:i, :])
        y_test.append(output_dataset[i])
        
    # Convert the data to a numpy array
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    return x_train, y_train, x_test, y_test, scaler

def create_graph_close(stock_name, company_list):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Close'].plot()
        plt.ylabel('Close Price')
        plt.xlabel(None)
        plt.title(f"Close Price of {stock_name[i - 1]}")
        
    plt.tight_layout()
    
    plt.savefig('images/graph_close_price.png') # Save graph as png file

def create_graph_delta(stock_name, company_list):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    apple = pd.DataFrame()
    apple['Delta_0'] = company_list[0]['Close']
    apple['Delta_1'] = apple['Delta_0'].diff(1)
    apple['Delta_2'] = apple['Delta_0'].diff(2)
    apple['Delta_3'] = apple['Delta_0'].diff(3)

    for i in range(4):
        plt.subplot(2, 2, i+1)
        apple[f'Delta_{i}'].plot()
        plt.ylabel(f'Close Price of Apple_delta_{i}')
        plt.xlabel(None)
        plt.title(f"Close Price of Apple_delta_{i}")
    
    plt.tight_layout()
    
    plt.savefig('images/graph_apple_delta.png') # Save graph as png file

def create_graph_correlation(stock_name, data_list, figure_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.concat([data_list[i]['Close'] for i in range(len(data_list))], axis=1).corr(), annot=True)
    plt.xticks(range(len(stock_name)), stock_name, rotation=45, ha='right')
    plt.yticks(range(len(stock_name)), stock_name, rotation=0)
    plt.title(f'{figure_name}')
    plt.savefig(f'images/{figure_name}.png')    
