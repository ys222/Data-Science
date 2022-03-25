# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:17:02 2020

@author: yashr
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
from lifelines import KaplanMeierFitter

data = pd.read_csv('Patient.csv')
data.head()
print(data.head())

print(data.columns)

data.info()
print(data.info())

data.describe()
print(data.describe())


print(data['Followup'].hist())



data.loc[data.Followup == 1, 'Eventtype'] = 0
data.loc[data.Followup == 0, 'Eventtype'] = 1

data.head()
print(data.head())

kmf = KaplanMeierFitter()
kmf.fit(durations = data["Followup"], event_observed = data["Eventtype"])


kmf.event_table
print(kmf.event_table)

event_at_0 = kmf.event_table.iloc[0,:]
surv_for_0 = (event_at_0.at_risk - event_at_0.observed)/event_at_0.at_risk
print(surv_for_0)


event_at_5 = kmf.event_table.iloc[2,:]
surv_for_5 = (event_at_5.at_risk - event_at_5.observed)/event_at_5.at_risk
print(surv_for_5)

print(kmf.survival_function_)


kmf.plot()
plt.xlable('Followup')
plt.ylable('Eventtype')
plt.title("KMF")




