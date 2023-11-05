from pandas_datareader import data as pdr
import yfinance as yf

def get_data():
    # Get data from Yahoo Finance
    yf.pdr_override()
    data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
    return data