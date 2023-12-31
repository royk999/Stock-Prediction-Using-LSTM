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

np.random.seed(15)

def single_improved_modify_df(df, training_dataset_percentage=0.8, validation_dataset_percentage=0.1, x_train_len=60):   
    len_data = len(df) - x_train_len + 1

    dataset = df.filter(['Close']).values
    output_dataset = dataset

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset)
    scaled_output_dataset = scaler.fit_transform(output_dataset.reshape(-1, 1)) # reshape(-1, 1) to convert 1D array to 2D array

    training_data_len = int(np.ceil((training_dataset_percentage) * len_data))
    validation_data_len = int(np.ceil((validation_dataset_percentage) * len_data))
    training_indexes = np.random.choice(training_data_len + validation_data_len, training_data_len, replace=False)

    train_val_x_raw_data = scaled_dataset[:training_data_len+validation_data_len+x_train_len-1, :]  
    train_val_y_raw_data = scaled_output_dataset[:training_data_len + validation_data_len, :]

    train_val_x_data = []
    train_val_y_data = []

    for i in range(training_data_len+validation_data_len):
        train_val_x_data.append(train_val_x_raw_data[i:i+x_train_len, :])
        train_val_y_data.append(train_val_y_raw_data[i, :])

    train_val_x_data, train_val_y_data = np.array(train_val_x_data), np.array(train_val_y_data)

    x_train = train_val_x_data[training_indexes, :, :]
    y_train = train_val_y_data[training_indexes, :]
    x_val = np.delete(train_val_x_data, training_indexes, axis=0)
    y_val = np.delete(train_val_y_data, training_indexes, axis=0)

    x_train, y_train = np.array(x_train), np.array(y_train)
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
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['MAPE'])

    early_stopper = EarlyStopping(monitor='loss', patience=5, verbose=0)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, callbacks=[early_stopper], validation_data=(x_val, y_val), shuffle=True)

    return model

def predict_single_improved_model(model, x_test, y_test, scalar):
    # Get the models predicted price values 
    predictions = model.predict(x_test) 

    predictions = scalar.inverse_transform(predictions) # Undo scaling
    y_test = scalar.inverse_transform(y_test) # Undo scaling
    return predictions, y_test

def analyze_single_improved(y_test, predictions):
    profit_model = 0
    profit_random = 0
    profit_always = 0
    
    sz = len(predictions)

    for i in range(0, sz-1):
        if predictions[i+1] - y_test[i] > 0:
            profit_model += y_test[i+1] - y_test[i]
        
        rand_int = np.random.randint(0, 10)
        if rand_int % 2 == 0:
            profit_random += y_test[i+1] - y_test[i]

        profit_always += y_test[i+1] - y_test[i]

    profit_model_rate = 100
    profit_random_rate = 100
    profit_always_rate = 100

    for i in range(0, sz-1):
        if predictions[i+1] - predictions[i] > 0:
            profit_model_rate *= y_test[i+1] / y_test[i]
        
        rand_int = np.random.randint(0, 10)
        if rand_int % 2 == 0:
            profit_random_rate *= y_test[i+1] / y_test[i]

        profit_always_rate *= y_test[i+1] / y_test[i]
    
    accuracy = 0.0

    for i in range(0, sz-1):
        if (predictions[i+1] - y_test[i]) * (y_test[i+1] - y_test[i]) > 0:
            accuracy += 1
    
    accuracy = accuracy / (sz-1)


    return profit_model, profit_random, profit_always, profit_model_rate, profit_random_rate, profit_always_rate, accuracy


def evaluate_single_improved(rmse, mape, path = 'results/results_single_improved_model.txt', features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs = 1, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    print(f'rmse: {rmse}, MAPE: {mape}')
    with open(path, 'a') as f:
        #f.write(f'rmse: {rmse}, MAPE: {mape} - features_lstm: {features_lstm}, feature_dense: {features_dense}, optimizer: {optimizer}, batch_size: {batch_size}, learning_rate: {learning_rate}, clipvalue: {clipvalue}\n')
        f.write(f'rmse: {rmse}, MAPE: {mape} - features_lstm: {features_lstm}, optimizer: {optimizer}, batch_size: {batch_size}, learning_rate: {learning_rate}\n')

def return_metrics_single_improved(y_test, predictions):
    RMSE = 0
    MAPE = 0

    sz = len(predictions)

    
    for i in range(sz):
        RMSE += (predictions[i] - y_test[i]) ** 2
        MAPE = MAPE + abs((predictions[i] - y_test[i]) / y_test[i])
    '''
        if(abs((predictions[i] - y_test[i]) / y_test[i]) > 0.1):
            print(f'index: {i}, prediction: {predictions[i]}, actual: {y_test[i]}')
    '''

    RMSE = np.sqrt(RMSE / sz)
    MAPE = MAPE * 100 / sz

    return RMSE, MAPE


def predict_singular(model, x_test, scaler):
    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

