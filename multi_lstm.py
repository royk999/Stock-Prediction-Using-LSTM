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

np.random.seed()

def multi_modify_df(stock_list, output_data, training_dataset_percentage, validation_dataset_percentage, x_train_len):   
    len_data = len(stock_list[0])

    data = pd.concat(stock_list, axis=1)
    dataset = data.values
    output_dataset = output_data.values

    training_data_len = int(np.ceil(len_data * training_dataset_percentage))
    validation_data_len = int(np.ceil(len_data * validation_dataset_percentage))

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset)
    scaled_output_dataset = scaler.fit_transform(output_dataset.reshape(-1, 1)) # reshape(-1, 1) to convert 1D array to 2D array
    training_x_data = scaled_dataset[0:training_data_len, :]
    training_y_data = scaled_output_dataset[0:training_data_len, :]

    # Replace NaN values with 0
    training_x_data = np.nan_to_num(training_x_data)
    training_y_data = np.nan_to_num(training_y_data)

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

    # Replace NaN values with 0
    x_val = np.nan_to_num(x_val)
    y_val = np.nan_to_num(y_val)

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
        
    # Replace NaN values with 0
    x_test = np.nan_to_num(x_test)
    y_test = np.nan_to_num(y_test)

    # Convert the data to a numpy array
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    
        
    return x_train, y_train, x_val, y_val, x_test, y_test, scaler


def multi_model_train(x_train, y_train, x_val, y_val, features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs=20, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
        print("NaN values found in training data. Please clean your data.")
        nan_indices = np.where(np.isnan(x_train))
        print(nan_indices)
        return None
    if np.any(np.isinf(x_train)) or np.any(np.isinf(y_train)):
        print("Inf values found in training data. Please clean your data.")
        return None

    model = Sequential()
    model.add(LSTM(features_lstm, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['MAPE'])

    early_stopper = EarlyStopping(monitor='loss', patience=10, verbose=1)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, callbacks=[early_stopper], validation_data=(x_val, y_val), shuffle=True)

    return model

def predict_multi_model(model, x_test, y_test, scalar):
    # Get the models predicted price values 
    predictions = model.predict(x_test) 
    print(f'predictions.shape = {predictions.shape}')

    predictions = scalar.inverse_transform(predictions) # Undo scaling
    y_test = scalar.inverse_transform(y_test) # Undo scaling
    return predictions, y_test

def return_metrics_multi(y_test, predictions, features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs = 1, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    rmse = 0
    MAPE = 0

    sz = len(predictions)
    for i in range(sz):
        rmse += (predictions[i] - y_test[i]) ** 2
        MAPE = MAPE + abs((predictions[i] - y_test[i]) / y_test[i])
    
    rmse = np.sqrt(rmse / sz)

    return rmse, MAPE

def evaluate_multi_model(rmse, mape, path = 'results/results_multi_model.txt', features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs = 1, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    print(f'rmse: {rmse}, MAPE: {mape}')
    with open(path, 'a') as f:
        #f.write(f'rmse: {rmse}, MAPE: {mape} - features_lstm: {features_lstm}, feature_dense: {features_dense}, optimizer: {optimizer}, batch_size: {batch_size}, learning_rate: {learning_rate}, clipvalue: {clipvalue}\n')
        f.write(f'rmse: {rmse}, MAPE: {mape} - optimizer: {optimizer}, batch_size: {batch_size}, learning_rate: {learning_rate}\n')