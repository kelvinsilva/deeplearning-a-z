# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:14:24 2018

@author: kelvi
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('./ml-1m/ml-1m/movies.dat', sep = '::', header = None,
                     engine = 'python', encoding = 'latin-1')
users = pd.read_csv('./ml-1m/ml-1m/users.dat', sep = '::', header = None,
                     engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('./ml-1m/ml-1m/ratings.dat', sep = '::', header = None,
                     engine = 'python', encoding = 'latin-1')

training_set = pd.read_csv('./ml-100k/ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('./ml-100k/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(training_set, dtype = 'int')

# getting number of users and movies

nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# 
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the neural network

#Stacked auto encoder
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) #20 nodes / features in first layer
        self.fc2 = nn.Linear(20, 10) #full connection 2
        self.fc3 = nn.Linear(10, 20) #third fullconnection
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):   # forward feed into autoencoder
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    

sae = SAE()
criterion = nn.MSELoss() # loss measure
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

        
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) #obtain the first batch of data
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input) # call the forward method
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target) # measure the loss
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #get the average of rated movies error
            loss.backward() # determine which direction to update weight
            train_loss += np.sqrt(loss.data[0]*mean_corrector) # keep running count of training loss
            s += 1.     
            optimizer.step() # how much to update the weights
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


#testing the SAE
    
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) #obtain the first batch of data
    target = input.clone()
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) #obtain the first batch of data
        target = input.clone()
        if torch.sum(target.data > 0) > 0: #make sure at least one rating
            output = sae(input) # call the forward method
            target.require_grad = False
            output[target == 0] = 0 #ignore non ratings
            loss = criterion(output, target) # measure the loss
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #get the average of rated movies error
            loss.backward() # determine which direction to update weight
            train_loss += np.sqrt(loss.data[0]*mean_corrector) # keep running count of training loss
            s += 1.     
            optimizer.step() # how much to update the weights
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))





