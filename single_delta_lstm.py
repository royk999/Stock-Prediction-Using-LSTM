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
from keras.callbacks import EarlyStopping


def modify_df_single_delta(df, training_dataset_percentage, x_train_len):
    close_values = df.filter(['Close']).values

    size_close_values = close_values.shape[0]

    dataset = np.zeros(size_close_values)

    for i in range(1, size_close_values):
        if(close_values[i - 1] == 0):
            dataset[i] = 0
            print(f'close_values[{i - 1}] == 0')
        else:
            dataset[i] = (close_values[i] - close_values[i - 1]) / close_values[i - 1]
    
    dataset = dataset.reshape(-1, 1)
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
        y_train.append(dataset[i])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = []
    for i in range(x_train_len, len(test_data)):
        x_test.append(test_data[i-x_train_len:i, 0])
        y_test.append(dataset[i])
    
    # Convert the data to a numpy array
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    return x_train, y_train, x_test, y_test, scaler

def single_delta_model_train(x_train, y_train, features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs=20, batch_size=1, learning_rate=0.001, clipvalue=1.0):
    if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
        print("NaN values found in training data. Please clean your data.")
        return None
    if np.any(np.isinf(x_train)) or np.any(np.isinf(y_train)):
        print("Inf values found in training data. Please clean your data.")
        return None

    model = Sequential()
    model.add(LSTM(features_lstm, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(features_dense, activation = "relu"))
    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stopper = EarlyStopping(monitor='loss', patience=5, verbose=1)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, callbacks=[early_stopper])

    return model

def predict_single_delta(model, x_test, scaler):
    # Get the models predicted price values 
    predictions = model.predict(x_test) 
    print(f'predictions: {predictions[:10]}')
    return predictions

def analyze_single_delta(y_test, predictions, scalar, features_lstm = 128, features_dense = 25, optimizer = 'Adam', max_epochs = 1, batch_size=1, learning_rate=0.001, clipvalue=1.0): 
    print(f'y_test: {y_test[:10]}')
    rmse = 0
    accuracy = 0
    
    sz = len(predictions)
    for i in range(sz):
        rmse += (predictions[i] - y_test[i]) ** 2
        if predictions[i] > 0 and y_test[i] > 0:
            accuracy += 1
        elif predictions[i] < 0 and y_test[i] < 0:
            accuracy += 1
    rmse = np.sqrt(rmse / sz)
    accuracy = accuracy / sz

    print(f'Root Mean Square Error: {rmse}')
    print(f'Accuracy: {accuracy}')
    #write rmse to file named results_single_delta_model.txt in results folder
    with open('results/results_single_delta_model.txt', 'a') as f:
        f.write(f'rmse: {rmse}, accuracy:{accuracy} - features_lstm: {features_lstm}, feature_dense: {features_dense}, optimizer: {optimizer}, batch_size: {batch_size}, learning_rate: {learning_rate}\n')

    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Single Delta LSTM Model')
    plt.xlabel('Date from 2020-10-30 (days)')
    plt.ylabel('Close Price ($)')    
    plt.savefig('images/results_single_delta_model_predicting_delta(1).png')
