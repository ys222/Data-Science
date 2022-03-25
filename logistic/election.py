# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:05:05 2020

@author: yashr
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

data = pd.read_csv("election_data.csv",sep=';')
print(data.head())

print(data.head())

data = data.dropna()
print(data.shape)
print(list(data.columns))

print(data.isnull().sum())

X = data.iloc[:, :1].values
y = data.iloc[:, 9].values
# evaluate the model by splitting the data-set into train and test sets
X_train, X_test, y_train, y_test = split(X, y, test_size=0.3)

model2 = LogisticRegression()
model2.fit(X_train, y_train)


predicted = model2.predict(X_test)
print(y_test)
predicted


# generate class probabilities
probs = model2.predict_proba(X_test)
probs


# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))

import seaborn as sns
conf_matrix = metrics.confusion_matrix(y_test, predicted)
sns.heatmap(conf_matrix, annot=True,cmap='Blues')

print(metrics.classification_report(y_test, predicted))