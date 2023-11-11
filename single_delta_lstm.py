import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np 
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler # for data scaling
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM


def modify_df_single_delta(df, training_dataset_percentage, x_train_len):
    # Create a new dataframe with only the 'Close' column
    dataset = df.filter(['Close']).diff(1).values
    print(f'dataset shape: {dataset.shape}')
    # print where nan values are
    print(f'index of nan value: {np.argwhere(np.isnan(dataset))}')
    # set nan values as 0
    dataset = np.nan_to_num(dataset)

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

def single_delta_model_train(x_train, y_train, features_lstm = 128, features_dense = 25, optimizer = 'Adam', epochs=1, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
        print("NaN values found in training data. Please clean your data.")
        return None
    if np.any(np.isinf(x_train)) or np.any(np.isinf(y_train)):
        print("Inf values found in training data. Please clean your data.")
        return None

    model = Sequential()
    model.add(LSTM(features_lstm, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(25, return_sequences=False))
    model.add(Dense(features_dense))
    model.add(Dense(1))
    

    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    return model


def predict_single_delta(model, x_test, scaler):
    # Get the models predicted price values 
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions) # Undo scaling

    return predictions

def analyze_single_delta(y_test, predictions, features_lstm = 128, features_dense = 25, optimizer = 'Adam', epochs=1, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    rmse = 0
    sz = len(predictions)
    for i in range(sz):
        rmse += (predictions[i] - y_test[i]) ** 2
    rmse = np.sqrt(rmse / sz)

    print(f'Root Mean Square Error: {rmse}')
    #write rmse to file named results_single_delta_model.txt in results folder
    with open('results/results_single_delta_model.txt', 'a') as f:
        f.write(f'rmse: {rmse} - features_lstm: {features_lstm}, feature_dense: {features_dense}, optimizer: {optimizer}, epochs: {epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}, clipvalue: {clipvalue}\n')

    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Single Delta LSTM Model')
    plt.xlabel('Date from 2020-10-30 (days)')
    plt.ylabel('Close Price ($)')    
    plt.savefig('images/results_single_delta_model_predicting_delta(1).png')
