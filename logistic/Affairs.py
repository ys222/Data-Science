# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:22:49 2020

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

# load dataset
dta = sm.datasets.fair.load_pandas().data

# adding "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)
dta = dta.rename(columns={"rate_marriage": "rateMarriage", "yrs_married": "yearsMarried","occupation_husb":"husbandOccupation"})

print(dta.sample(5))


print(dta.groupby('affair').mean())

print(dta.groupby('rateMarriage').mean())

# histogram of education
dta.educ.hist()
plt.title('Histogram of Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')


# histogram of marriage rating
dta.rateMarriage.hist()
plt.title('Histogram of Marriage Rating')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')


pd.crosstab(dta.rateMarriage, dta.affair.astype(bool)).plot(kind='bar')
plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')


affair_yrs_married = pd.crosstab(dta.yearsMarried, dta.affair.astype(bool))
affair_yrs_married.div(affair_yrs_married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Affair Percentage by Years Married')
plt.xlabel('Years Married')
plt.ylabel('Percentage')


# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('affair ~ rateMarriage + age + yearsMarried + children + \
                  religious + educ + C(occupation) + C(husbandOccupation)',
                  dta, return_type="dataframe")
X.columns
print(X.head(5))


# fix column names of X
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(husbandOccupation)[T.2.0]':'occ_husb_2',
                        'C(husbandOccupation)[T.3.0]':'occ_husb_3',
                        'C(husbandOccupation)[T.4.0]':'occ_husb_4',
                        'C(husbandOccupation)[T.5.0]':'occ_husb_5',
                        'C(husbandOccupation)[T.6.0]':'occ_husb_6'})



print(X.head())


# flatten y into a 1-D array
y = np.ravel(y)


model = LogisticRegression()
model = model.fit(X, y)

#accuracy obtained from training dataset
model.score(X, y)


# what percentage had affairs?
print(y.mean())


# examine the coefficients
X.columns, np.transpose(model.coef_)


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



