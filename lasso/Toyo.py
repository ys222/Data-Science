# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:46:06 2020

@author: yashr
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('ToyotaCorolla.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

X = X[:, 1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(y_pred)

print(y_test)

# load and summarize the housing dataset
from pandas import read_csv
from matplotlib import pyplot

dataset = pd.read_csv('ToyotaCorolla.csv')

# summarize shape
print(dataset.shape)
# summarize first few lines
print(dataset.head())



# define model
model = Ridge(alpha=1.0)
dataset = dataset.values
X, y = dataset[:, :-1], dataset[:, -1]
# define model
model = Ridge(alpha=1.0)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))