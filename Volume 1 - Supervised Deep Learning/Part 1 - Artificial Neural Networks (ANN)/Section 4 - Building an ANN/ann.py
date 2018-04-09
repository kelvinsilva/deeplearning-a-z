# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()   # encode country category
x[:, 1] = labelencoder_X_1.fit_transform(x[:,1])
labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1]) # one hot encode
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:] # avoid dummy variable trap


# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Part 2 - Artificial Neural Network

# importing the keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the neural network
classifier = Sequential()

# Adding hidden layer and input layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11 ))

# Add second hidden layer
classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))

# Compile the neural network
# Do stochastic gradient descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Fit the ann to the training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Make predictions

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# homework challenge test
custom_test = np.array([[0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000] ])
custom_test = sc_x.transform(custom_test)
a = classifier.predict(custom_test)

# Part 4 - Evaluate the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11 ))
    classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = 1)






