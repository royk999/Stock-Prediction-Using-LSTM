from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np 
import seaborn as sns
from datetime import datetime

def get_data(stock_list, start, end):
    # Get data from Yahoo Finance
    yf.pdr_override()

    data = {}

    for stock in stock_list:
        data[stock] = yf.download(stock, start, end)
    

    company_list = [data['AAPL'], data['MSFT'], data['GOOG'], data['AMZN']]
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name
        
    df = pd.concat(company_list, axis=0)
    return df


'''def create_graph_high():
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['High'].plot()
        plt.ylabel('High Price')
        plt.xlabel(None)
        plt.title(f"High Price of {tech_list[i - 1]}")
        
    plt.tight_layout()
'''
