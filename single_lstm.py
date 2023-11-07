from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np 
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler # for data scaling
from keras.models import Sequential
from keras.layers import Dense, LSTM

def get_df_singular():
    # Get data from Yahoo Finance
    yf.pdr_override()

    data = {}

    stock = 'AAPL'
    company_name = 'APPLE'
    end = datetime(2022, 10, 30)
    start = datetime(2022, 5, 30)

    data[stock] = yf.download(stock, start, end)
    

    company_list = [data['AAPL']]

    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name
        
    df = pd.concat(company_list, axis=0)
    return df



def modify_df_singular(df, training_dataset_percentage, x_train_len):
    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * training_dataset_percentage))

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    training_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(x_train_len, training_data_len):
        x_train.append(training_data[i - x_train_len:i, 0])
        y_train.append(training_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    return x_train, y_train, x_test, y_test


def single_lstm_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
 
    return model






