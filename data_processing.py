from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler # for data scaling
import yfinance as yf

def get_df(stock_list, company_name, start, end):
    # Get data from Yahoo Finance
    yf.pdr_override()

    data = {}

    for stock in stock_list:
        data[stock] = yf.download(stock, start, end)
    

    company_list = [data['AAPL'], data['MSFT'], data['GOOG'], data['AMZN']]

    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name
        
    df = pd.concat(company_list, axis=0)
    return (df, company_list)

def create_graph_close(company_name, company_list):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Close'].plot()
        plt.ylabel('Close Price')
        plt.xlabel(None)
        plt.title(f"Close Price of {company_name[i - 1]}")
        
    plt.tight_layout()
    
    plt.savefig('graph_close_price.png') # Save graph as png file

def create_graph_delta(company_name, company_list):
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
    
    plt.savefig('graph_apple_delta.png') # Save graph as png file

    
def modify_df(df, training_dataset_percentage):
    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * training_dataset_percentage))

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    return scaled_data, training_data_len