# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:03:36 2018

@author: kelvi
"""

#unsupervised learning
# Self organizing map. Fraud detector
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# MID mean inter neuron distance
#higher mid outlier

#visualize
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markeredgecolor = colors[y[i]],
         markerfacecolor = 'None', markersize = 10, markeredgewidth = 2)
    
show()
    

mappings = som.win_map(X)
frauds = np.concatenate( (mappings[(5,3)], mappings[(8, 3)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# part 2, use supervised learning to learn the probability of fraud
# create matrix of features

customers =  dataset.iloc[:, 1:].values
is_fraud = np.zeros(len(dataset)) #Initializde w zeros
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# importing the keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the neural network
classifier = Sequential()

# Adding hidden layer and input layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15 ))

# Adding the output layer
classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the neural network
# Do stochastic gradient descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Fit the ann to the training set
classifier.fit(customers, is_fraud, batch_size = 1, nb_epoch = 2)

# Add dropout layers: keras.layers.Dropout(rate, noise_shape=None, seed=None)

# Part 3 - Make predictions

y_pred = classifier.predict(customers) #probability customer cheated
y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred) , axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()] #sorted list of customers with highest probability of fraud

