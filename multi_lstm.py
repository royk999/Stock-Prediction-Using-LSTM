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

def multi_model_train(x_train, y_train, features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs=20, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
        print("NaN values found in training data. Please clean your data.")
        return None
    if np.any(np.isinf(x_train)) or np.any(np.isinf(y_train)):
        print("Inf values found in training data. Please clean your data.")
        return None

    model = Sequential()
    model.add(LSTM(features_lstm, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(features_dense))
    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stopper = EarlyStopping(monitor='loss', patience=5, verbose=1)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, callbacks=[early_stopper])

    return model

def predict_multi_model(model, x_test, scaler):
    # Get the models predicted price values 
    predictions = model.predict(x_test) 

    return predictions

def analyze_multi(y_test, predictions, features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs = 1, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    rmse = 0
    sz = len(predictions)
    for i in range(sz):
        rmse += (predictions[i] - y_test[i]) ** 2
    rmse = np.sqrt(rmse / sz)

    print(f'Root Mean Square Error: {rmse}')

    with open('results/results_multi_model.txt', 'a') as f:
        f.write(f'rmse: {rmse} - features_lstm: {features_lstm}, feature_dense: {features_dense}, optimizer: {optimizer}, batch_size: {batch_size}, learning_rate: {learning_rate}, clipvalue: {clipvalue}\n')

    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Multi LSTM Model')
    plt.xlabel('Date from 2020-10-30 (days)')
    plt.ylabel('Close Price ($)')    
    plt.savefig('images/results_multi_model_predicting_delta(1).png')
