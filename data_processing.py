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

    
def modify_df(company_list, training_dataset_percentage):
    # Create a new dataframe with only the 'Close' column
    dataset = []

    for company in company_list:
        dataset.append(company.filter(['Close']).values)
    
    training_data_len = int(np.ceil(len(dataset[0]) * training_dataset_percentage))
    
    print(f"training_data_len: {training_data_len}")


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


def create_graph_correlation(stock_name, company_list):
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.concat([company_list[i]['Close'] for i in range(len(company_list))], axis=1).corr(), annot=True)
    plt.title('Correlation between Companies')
    plt.xticks(range(len(stock_name)), stock_name)
    plt.yticks(range(len(stock_name)), stock_name)
    plt.savefig('images/graph_correlation.png')    

    
    
    