# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:47:15 2017

@author: j.klen
"""

# this script loads xgboost model from disk and returns array of predictions on loaded csv dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('diamondsWithNAinNums.csv')
droped_rows_index = data[data.isnull().any(axis = 1)].index
dataset = data.dropna()
dataset = dataset.values # to numpy array

class_var_index = 1 # index of variable to predict

# split the dataset into target variable array (y) and input variables (x)

x = np.delete(dataset, class_var_index, axis = 1)
y = dataset[:,class_var_index]

# encode target string column (y) to integers

label_encoder_y = LabelEncoder()
label_encoder_y.classes_ = np.load('label_classes.npy')
label_encoded_y = label_encoder_y.transform(y)

# integer and one hot encoding of string input variables

encoded_x = False
vars_to_drop = []

for i in range(0, X.shape[1]):
    if type(X[1,i]) == str:
        vars_to_drop.append(i)
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(X[:,i])
        feature = feature.reshape(X.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse = False)
        feature = onehot_encoder.fit_transform(feature)
        feature = np.delete(feature, feature.shape[1] - 1, axis = 1) # drop one dummy category of variable to encode
        if encoded_x == False:
            encoded_x_var = feature
            encoded_x = True
        else:
            encoded_x_var = np.concatenate((encoded_x_var, feature), axis = 1)

x = np.delete(x, vars_to_drop, axis = 1)
x = np.concatenate((x, encoded_x_var), axis = 1)

# load model from file

model = pickle.load(open('xgbModel.dat', 'rb'))

y_pred = model.predict(x)
predictions = label_encoder_y.inverse_transform(y_pred)

for i in range(0, len(droped_rows_index)):
    predictions = np.insert(predictions, droped_rows_index[i], np.nan)

predictions # numpy array of predicted classes

accuracy = accuracy_score(label_encoded_y, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

