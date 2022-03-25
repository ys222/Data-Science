# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:58:23 2020

@author: yashr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import folium
#%matplotlib inline
plt.style.use('seaborn-whitegrid')

data = pd.read_csv('Q1_a.csv')
data.shape
print(data.shape)

data.describe()
print(data.describe())

data.info()
print(data.info())


##data acess

data.columns
print(data.columns)

len(data)
print(len(data))

data.Index.unique()
print(data.Index.unique())

data.speed.unique()
print(data.speed.unique())

data.dist.unique()
print(data.dist.unique())

#count the value.
data.Index.value_counts()
print(data.Index.value_counts())

data.speed.value_counts()
print(data.speed.value_counts())

data.dist.value_counts()
print(data.dist.value_counts())

data.head(3)
print(data.head(3))

data.tail(15)
print(data.tail(15))

#find the argsort value
data = np.argsort(data , axis = 1)
print(data)



print(data)

from scipy.stats import skew
skew(data)

print(skew(data))

# skewness along the index axis 
data.skew(axis = 0, skipna = True) 

#data.hist(alpha=0.5, figsize=(16, 10))

sns.kdeplot(data.query('speed < 78').speed)
sns.kdeplot(data.query('speed < 56').speed)

from scipy.stats import norm, kurtosis
kurtosis(data)
print(kurtosis(data))

import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis
x = np.linspace(-5, 5, 100)
ax = plt.subplot()
distnames = ['laplace', 'norm', 'uniform']
for distname in distnames:
    if distname == 'uniform':
        dist = getattr(stats, distname)(loc=-2, scale=4)
    else:
        dist = getattr(stats, distname)
    data = dist.rvs(size=1000)
    kur = kurtosis(data, fisher=True)
    y = dist.pdf(x)
    ax.plot(x, y, label="{}, {}".format(distname, round(kur, 3)))
    ax.legend()