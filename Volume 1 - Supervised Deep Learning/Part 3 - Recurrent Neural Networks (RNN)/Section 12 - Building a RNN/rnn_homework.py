# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 15:25:27 2018

@author: kelvi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")

#create numpy array
training_set = dataset_train.iloc[:, 1:2].values #make correct dimension of stock price

#Feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))   # Create the scaler object
training_set_scaled = sc.fit_transform(training_set) # use the scaler to scale our data

# create a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train,), np.array(y_train)

#Reshaping , add new dimensions or indicators for stk price
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 1000, return_sequences = True, input_shape = (X_train.shape[1], 1) ) )
regressor.add(Dropout(0.2))

# Add more layers
regressor.add(LSTM(units = 1000, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 1000, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 1000)) # For last layer dont need return sequence
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

# Compile the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train NN
regressor.fit(X_train, y_train, epochs = 100, batch_size = 64)



# Part 3 making predictions and visualizing results
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Gettin ght predicted stock price 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Vizualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()







