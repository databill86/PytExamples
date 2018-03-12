# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:52:16 2017

@author: j.klen
"""

# done in python 3.5
# drops rows which contain missing value in any of the columns
# works also with string variables, which are one-hot encoded
# returns integer numpy array, which contains: -1 as outlier, 1 as inlier, 0 as NA value (rows dropped from input DF)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import LocalOutlierFactor

#data = pd.read_csv('diamonds.csv')
#data = pd.read_csv('diamondsWithNAinStrings.csv')
data = pd.read_csv('diamondsWithNAinNums.csv')

droped_rows_index = data[data.isnull().any(axis = 1)].index

dataset = data.dropna()
dataset = dataset.values # to numpy array

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

x = np.delete(dataset, vars_to_drop, axis = 1)
x = np.concatenate((x, encoded_x_var), axis = 1)

clf = LocalOutlierFactor(n_neighbors=20) # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
outliers = clf.fit_predict(x)

for i in range(0, len(droped_rows_index)):
    outliers = np.insert(outliers, droped_rows_index[i], 0)

outliers  # -1 outlier, 1 inlier, 0 NA