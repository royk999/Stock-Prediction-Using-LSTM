# Stock-Prediction-Using-LSTM

This project aims to improve the LSTM model which predicts Apple's Closing Price and ultimately creating a model which can decide how to invest on stocks. 

# Modeling Approach
This project is based on the kaggle's "stock market analysis & prediction using LSTM" project. From this Kaggle's model, this project ultimately aims to create a model with multiple-features inputs which predicts the stock value difference. 

# Data Collection
Apple Closing Price, delta(1,2,3)
Apple Opening Price, Closing Price - Opening Price
MADA
Nasdaq
CBOE Volatility Index (VIX)

# Hyperparameter Tuning
This project compared rmse errors to find the best hyperparameters that work the best. First, hyperparameters for training process was tuned. Then, the parameters for the neural network itself was tuned. 

## Training Process
Optimizers, Learning rates, batch sizes, 

## Neural Network

# Analyzing & evaluating the model
This project uses RMSE, MAPE, and R to evaluate the model. Moreover, this project aims to create an algorithm which makes an investment using the predictions of the model. Then, a model will be compared using the outcome of the investment. 

# References
https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm
