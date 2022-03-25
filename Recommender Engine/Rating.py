# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:13:07 2020

@author: yashr
"""


import numpy as np
import pandas as pd

ratings_data = pd.read_excel("Ratings.xlsx")
ratings_data.head()
print(ratings_data.head())


ratings_data.groupby('joke_id')['Rating'].mean().head()
print(ratings_data.groupby('joke_id')['Rating'].mean().head())

ratings_data.groupby('joke_id')['Rating'].mean().sort_values(ascending=False).head()
print(ratings_data.groupby('joke_id')['Rating'].mean().sort_values(ascending=False).head())


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
#%matplotlib inline

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_data['Rating'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_data['jike_id'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='joke_id', data=ratings_data, alpha=0.4)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import pandas as pd
data = pd.read_csv("Ratings.csv",index_col="JokeId")
data.head()
print(data.head())       

data = data.iloc[:,:5000]


from sklearn.metrics.pairwise import cosine_similarity
data = data.T
Filtering_cosim = cosine_similarity(data,data)
sums_of_columns = data.sum(axis=1)
columns_size = len(data.columns)
value = sums_of_columns/columns_size
index_of_max = value[value == value.max()].index[0]
print("The best joke is index as {} and value of joke is :{}".format(index_of_max,value.max()))


most_sim_users = sorted(list(enumerate(Filtering_cosim[8])), key=lambda x: x[1], reverse=True)
most_sim_users = most_sim_users[1:11]
sim_users = [x[0] for x in most_sim_users]
print(sim_users)


candidates_jokes = data.iloc[sim_users,:]

def UBCF(user_num):
    ### finding most similar users among matrix

    most_sim_users = sorted(list(enumerate(Filtering_cosim[user_num])), key=lambda x: x[1], reverse=True)
    most_sim_users = most_sim_users[1:11]

    ### user index and their similairity values 

    sim_users = [x[0] for x in most_sim_users]
    sim_values = [x[1] for x in most_sim_users]

    ### among users having most similar preferences, finding movies having highest average score
    ### however except the movie that original user didn't see

    candidates_jokes = data.iloc[sim_users,:]

    candidates_jokes.mean(axis=0).head()

    mean_score = pd.Series(candidates_jokes.mean(axis=0))
    mean_score = mean_score.sort_values(axis=0, ascending=False)
    

UBCF(100) 