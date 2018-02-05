# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:10:23 2017

@author: j.klen
"""

'current working directory'
import os
os.getcwd

'list files and folders in working directory'
os.listdir()

'type of object'
type(obj)

'sequence of numbers'
range(1,10,1)

'drop one/multiple columns from numpy array'
x = np.delete(x, 5, axis = 1) # or
x = np.delete(x, [1,3,4], axis = 1)

'merge 2d numpy arrays columnwise'
np.concatenate((x, encoded_x), axis = 1)

'save sklearn labelencoder classes'
np.save('label_classes.npy', label_encoder.classes_)

'dimensions of dataframe/object'
df.shape

'iterate over a sequence while keeping track of the item number'
for index, item in enumerate(words)

'list comprehensions'
[i**2 for i in range(10)]

print("accuracy: %s with %s souls" % ('kok', 'kok2'))
print("accuracy: %.3f%%" % 0.35)

'run script from IPython console'
%run xgboostTest.py

'reload modules'
reload(np) #python 2.7

import importlib #python 3
importlib.reload(pd)

"if __name__ == '__main__':" # this part is run only when module is run directly

'list of directiories  searched by Python for modules'
import sys
sys.path

'get module path'
pd.__file__

'running external command'
import os
os.system('dir')

'save and load arbitrary objects to/from file'
import pickle
pickle.dump(model, open('xgbModel.dat', 'wb'))

'measure execution time of small code snippets'
%timeit [i**2 for i in range(1000)] # or
import timeit
timeit.timeit('[i**2 for i in range(1000000)]', number = 1)

'ifelse equivalent like in R on numpy array'
np.where(outliers == -1, 'yes', 'no')

'get columns (true/false) of pandas dataframe, which contain some NA's'
data.isnull().any(axis = 0)

'get columns of pandas dataframe, which contain some NA's'
data.loc[:,data.isnull().any(axis = 0)]

'get rows of pandas dataframe, which contain some NA's'
data[data.isnull().any(axis = 1)]

'drop rows with NA values in pandas DF'
df.dropna() # drop row if any column contains NA
df.dropna(how = 'all') # drop row if all columns contain NA
df.dropna(thresh = 2) # drop row if it does not have at least two values that are not NA
df.dropna(subset = [1]) # drop row only if NA is present in specified column

'filtering pandas dataframe via column/row indexes'
dataset.iloc[0:4, 3:13]

'get index of float or integer columns in numpy array'
floats_or_ints_index = np.where([type(el) in [float, int] for el in dataset[0,:]])[0]

'count occurencies of items in numpy array' # https://ask.sagemath.org/question/10279/frequency-count-in-numpy-array/
from collections import Counter
Counter(clusters)