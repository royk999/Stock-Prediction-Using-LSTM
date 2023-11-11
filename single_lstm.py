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

def get_df_singular(stock_name, start, end):
    # Get data from Yahoo Finance
    yf.pdr_override()

    data = {}

    stock_name = 'AAPL'

    data[stock_name] = yf.download(stock_name, start, end)
    
    company_list = [data['AAPL']]
    company_list[0]["stock_name"] = stock_name
        
    df = pd.concat(company_list, axis=0)
    return df

def modify_df_singular(df, training_dataset_percentage, x_train_len):
    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    dataset = data.values

    print(f'type of dataset: {type(dataset)}')
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

    return x_train, y_train, x_test, y_test, scaler

def single_model_train(x_train, y_train):
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

def predict_singular(model, x_test, scaler):
    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

def analyze_singular(y_test, predictions):
    #turn y_test into a dataframe
    y_test = pd.DataFrame(y_test)
    predictions = pd.DataFrame(predictions)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)

    y_test_delta = y_test.diff(1)
    predictions_delta = predictions.diff(1)

    # change nan values to 0 
    y_test_delta = y_test_delta.fillna(0)
    predictions_delta = predictions_delta.fillna(0)

    sz = y_test_delta.shape[0]
    rmse_1 = 0.0
    for i in range(sz):
        rmse_1 = rmse_1 + (predictions_delta.iloc[i, 0] - y_test_delta.iloc[i, 0])**2

    rmse_1 = np.sqrt(rmse_1/sz)

    print(f'RMSE: {rmse}')
    print(f'RMSE_1: {rmse_1}')

    with open('results/results_single_model.txt', 'a') as f:
        f.write(f'RMSE_original: {rmse}\n')
        f.write(f'RMSE_delta_1: {rmse_1}')
    
    plt.plot(y_test_delta, label='Actual')
    plt.plot(predictions_delta, label='Predicted')
    plt.legend()
    plt.title(f'Single_Model_Delta(1)')
    plt.xlabel('Date from 2020-10-30 (days)')
    plt.ylabel('Close Price ($)')
    # Save graph as png file in a folder named images
    plt.savefig('images/results_single_model_delta(1).png')


