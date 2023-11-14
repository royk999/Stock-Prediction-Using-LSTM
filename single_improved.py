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
from keras.callbacks import EarlyStopping

def single_improved_modify_df(df, training_dataset_percentage, validation_dataset_percentage, x_train_len):   
    len_data = len(df)

    dataset = df.filter(['Close']).values
    output_dataset = dataset

    training_data_len = int(np.ceil(len_data * training_dataset_percentage))
    validation_data_len = int(np.ceil(len_data * validation_dataset_percentage))

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset)
    scaled_output_dataset = scaler.fit_transform(output_dataset.reshape(-1, 1)) # reshape(-1, 1) to convert 1D array to 2D array
    training_x_data = scaled_dataset[0:training_data_len, :]
    training_y_data = scaled_output_dataset[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(x_train_len, training_data_len):
        x_train.append(training_x_data[i - x_train_len:i, :])
        y_train.append(training_y_data[i, :])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Create the validation data set
    validation_x_data = scaled_dataset[training_data_len - x_train_len:training_data_len + validation_data_len, :]
    validation_y_data = scaled_output_dataset[training_data_len:training_data_len + validation_data_len, :]
    x_val = []
    y_val = []

    for i in range(len(validation_y_data)):
        x_val.append(validation_x_data[i:i+x_train_len, :])
        y_val.append(validation_y_data[i, :])

    # Convert the data to a numpy array
    x_val, y_val = np.array(x_val), np.array(y_val)

    # Create the testing data set
    test_x_data = scaled_dataset[training_data_len+validation_data_len - x_train_len: , :]
    test_y_data = scaled_output_dataset[training_data_len+validation_data_len:, :]
    x_test = []
    y_test = []

    for i in range(len(test_y_data)):
        x_test.append(test_x_data[i:i+x_train_len, :])
        y_test.append(test_y_data[i, :])
        
    # Convert the data to a numpy array
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, scaler


def single_improved_model_train(x_train, y_train, x_val, y_val, features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs=20, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
        print("NaN values found in training data. Please clean your data.")
        return None
    if np.any(np.isinf(x_train)) or np.any(np.isinf(y_train)):
        print("Inf values found in training data. Please clean your data.")
        return None

    model = Sequential()
    model.add(LSTM(features_lstm, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(features_dense, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['MAPE'])

    early_stopper = EarlyStopping(monitor='loss', patience=10, verbose=1)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, callbacks=[early_stopper], validation_data=(x_val, y_val), shuffle=True)

    return model

def predict_single_improved_model(model, x_test, y_test, scalar):
    # Get the models predicted price values 
    predictions = model.predict(x_test) 
    print(f'predictions.shape = {predictions.shape}')

    predictions = scalar.inverse_transform(predictions) # Undo scaling
    y_test = scalar.inverse_transform(y_test) # Undo scaling
    return predictions, y_test

def analyze_single_improved(y_test, predictions, features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs = 1, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    rmse = 0
    MAPE = 0

    sz = len(predictions)
    for i in range(sz):
        rmse += (predictions[i] - y_test[i]) ** 2
        MAPE = MAPE + abs((predictions[i] - y_test[i]) / y_test[i])
    
    rmse = np.sqrt(rmse / sz)
    MAPE = MAPE * 100 / sz

    print(f'Root Mean Square Error: {rmse}')
    print(f'MAPE: {MAPE}')
    with open('results/results_single_improved_model.txt', 'a') as f:
        f.write(f'rmse: {rmse}, MAPE: {MAPE} - features_lstm: {features_lstm}, feature_dense: {features_dense}, optimizer: {optimizer}, batch_size: {batch_size}, learning_rate: {learning_rate}, clipvalue: {clipvalue}\n')

    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Single_Improved Model')
    plt.xlabel('Date from 2020-10-30 (days)')
    plt.ylabel('Close Price ($)')    
    plt.savefig('images/results_single_improved_model.png')


def return_metrics_single_improved(y_test, predictions):
    RMSE = 0
    MAPE = 0

    sz = len(predictions)

    for i in range(sz):
        RMSE += (predictions[i] - y_test[i]) ** 2
        MAPE = MAPE + abs((predictions[i] - y_test[i]) / y_test[i])
    
    RMSE = np.sqrt(RMSE / sz)
    MAPE = MAPE / sz

    return RMSE, MAPE