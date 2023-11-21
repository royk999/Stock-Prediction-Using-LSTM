# GOAL

This project aims to improve the LSTM model which predicts Apple's Closing Price and ultimately creating a model which can decide how to invest on stocks. 

# Modeling Approach
This project ultimately aims to improve the model with multiple-features inputs which predicts the stock value. 
Basic model is the Kaggle model which uses single feature to predict the stock value. 
Single Improved model is the model that is improved from the basic model. I changed the number and types of the layers and also changed the optimizer, learning rates, and etc.
Multi model is the model which uses multiple features to predict the stock value. 

# Data Collection
Apple Closing Price, delta(1,2,3)
Apple Opening Price, Closing Price - Opening Price
MADA
Nasdaq
CBOE Volatility Index (VIX)
DX-Y.NYB (Dollar index)

# Hyperparameter Tuning
This project compared rmse errors to find the best hyperparameters that work the best. First, hyperparameters for training process was tuned. Then, the parameters for the neural network itself was tuned. 

## Training Process
Optimizers, Learning rates, batch sizes

## Neural Network
Number of 

# Analyzing & evaluating the model
This project uses RMSE, MAPE, and R to evaluate the model. 
Moreover, this project aims to create an algorithm which makes an investment using the predictions of the model. Then, a model will be compared using the outcome of the investment. 

# References
https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm



# 실험 과정

## 실패 과정
delta 값만을 가지고 학습한 결과 의미있는 예측을 하지 못했다. 가장 최적화를 시킨 모델이 x축에 평행한 어떤 값만을 예측했기 때문이다. 이는 delta값은 비교적 계절성을 띠지는 않지만 무작위적인 데이터이기 때문에 패턴을 추출하는 것이 불가능하기 때문이라고 판단했다.

## 목표의 변화
따라서 delta값을 가지고 학습하는 것은 포기했다. 대신, 새로운 두 가지 목표를 세웠다. 첫째로 인공지능 학습 과정과 인공지능 모델 자체를 튜닝해서 단변수 LSTM 모델을 최적화한다. 그리고 여러 가지 데이터를 추가해서 다변수 LSTM 모델을 구현한 뒤, 이 모델 역시 최적화한다.

## 단변수 LSTM 모델 최적화하기
현재 모델 구조: LSTM 한 층, Dense 2층 (논문에서 사용)

각 모델 구조(논문을 참고)를 기반으로 optimizer, learning_rate, batch_size를 정함
이를 정하기 위해서 각 hyperparameter에 대해서 10번 시행하고 평균 평가함수 값을 비교함 (RMSE, MAPE)
이때 max_epochs는 100으로, cutoff는 5로 설정

그 결과:
rmse: [1.19299093], MAPE: [1.26806332] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 1, learning_rate: 0.1, clipvalue: 1.0
rmse: [1.15517216], MAPE: [1.28326656] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 1, learning_rate: 0.01, clipvalue: 1.0
rmse: [1.26614123], MAPE: [1.54428161] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 1, learning_rate: 0.001, clipvalue: 1.0
rmse: [1.22435501], MAPE: [1.86984755] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 4, learning_rate: 0.1, clipvalue: 1.0
rmse: [1.18289782], MAPE: [1.52618809] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 4, learning_rate: 0.01, clipvalue: 1.0
rmse: [1.84776335], MAPE: [1.93069934] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 4, learning_rate: 0.001, clipvalue: 1.0
rmse: [1.22860832], MAPE: [1.89747907] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 8, learning_rate: 0.1, clipvalue: 1.0
rmse: [2.99947253], MAPE: [4.34703534] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 8, learning_rate: 0.01, clipvalue: 1.0
rmse: [1.70033211], MAPE: [4.44210574] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 8, learning_rate: 0.001, clipvalue: 1.0
rmse: [1.58260775], MAPE: [1.81417315] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 16, learning_rate: 0.1, clipvalue: 1.0
rmse: [1.29742636], MAPE: [2.29501322] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 16, learning_rate: 0.01, clipvalue: 1.0
rmse: [2.3558851], MAPE: [3.21428833] - features_lstm: 128, feature_dense: 25, optimizer: Adam, batch_size: 16, learning_rate: 0.001, clipvalue: 1.0

각 모델에 대해서 neuron의 수를 조정함

## 다변수 LSTM 모델 최적화하기





