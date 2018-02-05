# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:27:34 2018

@author: j.klen
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras # set "KERAS_BACKEND=tensorflow", in anaconda prompt when environment is activated
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from keras.wrappers.scikit_learn import KerasClassifier # makes possible to use keras models in scikit-learn
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# one hot encode and scale variables

LabelEncoder_Country = LabelEncoder()
LabelEncoder_Gender = LabelEncoder()

X[:,1] = LabelEncoder_Country.fit_transform(X[:,1])
X[:,2] = LabelEncoder_Gender.fit_transform(X[:,2])
X_cat = X[:,[1,2]]

OneHotEncoder = OneHotEncoder(sparse = False)
X_cat_encoded = OneHotEncoder.fit_transform(X_cat)

X = np.delete(X, [1,2], axis = 1)
StandardScaler = StandardScaler()
X_scaled = StandardScaler.fit_transform(X)
X = np.concatenate((X_scaled, X_cat_encoded), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# oversample to get balanced training dataset with RandomOverSampler

ros = RandomOverSampler(random_state = 0, ratio = {1:12000}) # oversample the 1-'exited' class to get model better towards it
X_train, y_train = ros.fit_sample(X_train, y_train)

'''
# oversample to get balanced training dataset with SMOTE
sm = SMOTE(random_state = 0)
X_train, y_train = sm.fit_sample(X_train, y_train)
'''
# initialising the ANN
classifier = Sequential() # defined as a sequence of layers. Other types?

# add input layer and first hidden layer
#classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 13)) # 13 input variables + 1 output node = 14/2, why?
classifier.add(Dense(input_dim = 13, activation = 'relu', units = 13, kernel_initializer = 'uniform'))
#classifier.add(Dropout(rate = 0.05))

# add 2nd  and 3rd hidden layer
classifier.add(Dense(activation = 'relu', units = 26, kernel_initializer = 'uniform'))
#classifier.add(Dropout(rate = 0.05)) # with this, the accuracy is higher by around 5% on test set, 8% higher on training set

classifier.add(Dense(activation = 'relu', units = 52, kernel_initializer = 'uniform')) # kernel_initializer - initialization of weights
classifier.add(Dropout(rate = 0.05))

# add output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# set optimizer

adam = optimizers.adam(lr = 0.001) # in .compile 'adam' may be used - with default parameters

# compiling the ANN
classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy']) # if more than 2 classes, 'categorical_crossentropy'

# fit the ANN to training set and put info about training into 'history'
history = classifier.fit(x = X_train, y = y_train, batch_size = 50, epochs = 500, validation_split = 0.2)

classifier.save('churn_keras_model1.h5')

# to check accuracy on train set
y_train_pred = classifier.predict(X_train)
y_train_pred = (y_train_pred > 0.5)

cm_train = confusion_matrix(y_train, y_train_pred)
print('train accuray')
print(cm_train[0,0]/(cm_train[0,1] + cm_train[0,0]))
print(cm_train[1,1]/(cm_train[1,0] + cm_train[1,1]))
print('overall train accuracy')
print((cm_train[0,0] + cm_train[1,1])/(cm_train[0,0] + cm_train[1,1] + cm_train[0,1] + cm_train[1,0]))

# predict and evaluate model on test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# making confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('test accuracy')
print(cm[0,0]/(cm[0,1] + cm[0,0]))
print(cm[1,1]/(cm[1,0] + cm[1,1]))
print('overall test accuracy')
print((cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1] + cm[0,1] + cm[1,0]))

# counts of target categories
print(Counter(Y))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# 1 customer test --------------------------------------------------------------------------

X_customer = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])


X_customer[:,1] = LabelEncoder_Country.transform(X_customer[:,1])
X_customer[:,2] = LabelEncoder_Gender.transform(X_customer[:,2])
X_customer_cat = X_customer[:,[1,2]]

X_customer_cat_encoded = OneHotEncoder.transform(X_customer_cat)

X_customer = np.delete(X_customer, [1,2], axis = 1)
X_customer_scaled = StandardScaler.transform(X_customer)
X_customer = np.concatenate((X_customer_scaled, X_customer_cat_encoded), axis = 1)

y_customer_pred = classifier.predict(X_customer)
y_customer_pred = (y_customer_pred > 0.5)
print('will this customer exit?')
print(y_customer_pred)

# evaluating accuracy of the ANN with cross validation -----------------------

def build_classifier():
    classifier = Sequential() # defined as a sequence of layers. Other types?
    classifier.add(Dense(input_dim = 13, activation = 'relu', units = 26, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(activation = 'relu', units = 26, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(activation = 'relu', units = 50, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier_keras = KerasClassifier(build_fn = build_classifier, batch_size = 50, epochs = 500)

# does not work  with n_jobs = -1, if __name__ == '__main__': should fix it, but still does not work, also after updating scikit-learn and keras
if __name__ == '__main__':
    accuracies = cross_val_score(estimator = classifier_keras, X = X_train, y = y_train, cv = 4, n_jobs = 1)
mean_accuracy = accuracies.mean()
std_accuracy = accuracies.std()

# classifier.predict(X_test) # does not work

# dropout regularization to reduce overfitting

# hyperparameter tuning with gridsearch -----------------------------------------------------

def build_classifier(optim_alg, dropout_rate):
    classifier = Sequential() # defined as a sequence of layers. Other types?
    classifier.add(Dense(input_dim = 13, activation = 'relu', units = 26, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = dropout_rate))
    classifier.add(Dense(activation = 'relu', units = 26, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = dropout_rate))
    classifier.add(Dense(activation = 'relu', units = 52, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = dropout_rate))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optim_alg, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[10, 50, 100], 'epochs':[100, 500], 'optim_alg':['adam'], 'dropout_rate':[0.05, 0.1, 0.15]} # 'rmsprop'

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 4, verbose = 10) # takes around 7 hours
grid_search_fitted = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
y_test_predicted = grid_search_fitted.best_estimator_.predict(X_test)

cm_test = confusion_matrix(y_test, y_test_predicted)
print('test accuracy - best model from grid search CV')
print(cm_test)
print(cm_test[0,0]/(cm[1,0] + cm_test[0,0]))
print(cm_test[1,1]/(cm[1,0] + cm_test[1,1]))

according_best_params = {'epochs':500, 'batch_size':50, 'dropout_rate':0.05, 'optim_alg':'adam'} # 0.8286 accuracy on validation set, 0.8357 4-fold CV test accuracy

