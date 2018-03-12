# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:24:37 2017

@author: j.klen
"""

# done in python 2.7
# installing xgboost on windows anaconda python 2.7 - conda install -c mndrake xgboost
# installing xgboost on windows anaconda python 3.5 - did not tried yet
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en

# this script creates  xgboost model of selected categorical variable and saves model to file

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('diamondsWithNAinNums.csv')
droped_rows_index = data[data.isnull().any(axis = 1)].index
dataset = data.dropna()
dataset = data.values # to numpy array

class_var_index = 1 # index of class variable to predict

# split the dataset into target variable array (y) and input variables (x)

x = np.delete(dataset, class_var_index, axis = 1)
y = dataset[:,class_var_index]

# encode target string column (y) to integers

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)
np.save('label_classes.npy', label_encoder.classes_)

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

seed = 6
test_size = 0.25
x_train, x_test, y_train, y_test = train_test_split(x, label_encoded_y, test_size = test_size, random_state = seed)
model = xgb.XGBClassifier()
model.fit(x_train, y_train)

# save model to file

pickle.dump(model, open('xgbModel.dat', 'wb'))
