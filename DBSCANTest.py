# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:28:07 2017

@author: j.klen
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# done in python 3.5
# performs density based clustering
# drops rows which contain missing value in any of the columns
# one-hot encodes string variables and scales integer and float variables
# returns string numpy array with cluster number where '-1' - noise, other numbers - clusters,
#   'nan' as NA value (rows dropped from input DF)

data = pd.read_csv('diamondsWithNAinNums.csv')

droped_rows_index = data[data.isnull().any(axis = 1)].index

dataset = data.dropna()
dataset = dataset.values # to numpy array

floats_or_ints_index = np.where([type(el) in [float, int] for el in dataset[0,:]])[0]
datasetPart_to_scale = dataset[:,floats_or_ints_index]
datasetPart_scaled = StandardScaler().fit_transform(datasetPart_to_scale)

encoded_x = False
vars_to_drop = []

for i in range(0, dataset.shape[1]):
    if type(dataset[1,i]) == str:
        vars_to_drop.append(i)
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(dataset[:, i])
        feature = feature.reshape(dataset.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse = False)
        feature = onehot_encoder.fit_transform(feature)
        if encoded_x == False:
            encoded_x_var = feature
        else:
            encoded_x_var = np.concatenate((encoded_x_var, feature), axis = 1)

#x = np.delete(dataset, vars_to_drop, axis = 1)
x = np.concatenate((datasetPart_scaled, encoded_x_var), axis = 1)

db = DBSCAN(min_samples = 100, eps = 0.9).fit(x) # change min_samples relatively according to size of input dataset
clusters = db.labels_
clusters = clusters.astype(str)

for i in range(0, len(droped_rows_index)):
    clusters = np.insert(clusters, droped_rows_index[i], np.NaN)

clusters