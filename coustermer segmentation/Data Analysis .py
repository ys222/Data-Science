# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:11:53 2020

@author: yashr
"""


import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff

data = pd.read_csv('Mall_Customers.csv')

data.head()
print(data.head())
data.tail()
print(data.tail())


data.isnull().any().any()
print(data.isnull().any().any())

data.shape
print(data.shape)
data.describe()
print(data.describe)
data.info()
print(data.info())

data.columns
print(data.columns)
len(data)
print(len(data))

# access a row with data
data.iloc[1]
print(data.iloc[1])
data.iloc[2]
print(data.iloc[2])
data.iloc[3]
print(data.iloc[3])
data.iloc[4]
print(data.iloc[4])
data.iloc[5]
print(data.iloc[5])

#pass the multiple value
data.iloc[[1 , 2 , 3 , 4 , 5]]
print(data.iloc[[1 , 2 , 3 , 4 , 5]])
data.iloc[[6 , 7 , 8 , 9 , 10]]
print(data.iloc[[6 , 7 , 8 , 9 , 10]])

# plot with pandas of hexbin plot/barplot/areaplot/lineplot 
data[data['Age'] < 100].plot.hexbin(x = 'Age' , y = 'CustomerID' , gridsize=15)

data[data['Age'] < 79].plot.bar(x = 'Age' , y = 'CustomerID' , stacked = True)

data[data['Age'] < 79].plot.area(x = 'Age' , y = 'CustomerID' , stacked = True)

data[data['Age'] < 100].plot.line(x = 'Age' , y = 'CustomerID' , stacked = True)

sns.jointplot(x = 'Age' , y = 'CustomerID' , data = data[data['CustomerID'] < 100])

plt.rcParams['figure.figsize'] = (15, 10)


plotting.andrews_curves(data.drop("CustomerID", axis=1), "Genre")
plt.title('Andrew Curves for Genre', fontsize = 20)
plt.show()

labels = ['Female', 'Male']
size = data['Genre'].value_counts()
colors = ['Red', 'Yellow']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Genre', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()

plt.rcParams['figure.figsize'] = (15, 8)
sns.countplot(data['Age'], palette = 'hsv')
plt.title('Distribution of Age', fontsize = 20)
plt.show()

sns.pairplot(data)
plt.title('Pairplot for the Data', fontsize = 20)
plt.show()

#plot lmplot of age with ID
sns.lmplot(x="Age", y="CustomerID", data=data)
plt.show()

#plot the subplot of the data
fig, axarr = plt.subplots(2, 1, figsize=(12, 8))
data['Age'].value_counts().sort_index().plot.bar(ax = axarr[0])
data['CustomerID'].value_counts().sort_index().plot.bar(ax = axarr[1])

#plot the stripplot
sns.stripplot(x="Age", y="CustomerID", data=data, jitter=True)

#plot scatter graph of Age with ID
data.plot.scatter(x = 'Age' , y = 'CustomerID' , title = 'Age per Customer')

#plot the line graph of black with ID
data.plot.line(x = 'Age' , y = 'CustomerID' , title = 'Age per Customer')

#plot the box graph of black with ID
data.plot.box(x = 'Age' , y = 'CustomerID' , title = 'Age per Customer')

#count/sum/mean of the value age with ID
data[['Age' , 'CustomerID']].groupby('Age').count()
print(data[['Age' , 'CustomerID']].groupby('Age').count())
data[['Age' , 'CustomerID']].groupby('Age').sum()
print(data[['Age' , 'CustomerID']].groupby('Age').sum())
data[['Age' , 'CustomerID']].groupby('Age').mean()
print(data[['Age' , 'CustomerID']].groupby('Age').mean())

#Cluster Plot between Genre vs Age
plt.rcParams['figure.figsize'] = (18, 7)
sns.stripplot(data['Genre'], data['Age'], palette = 'Purples', size = 10)
plt.title('Genre vs Age', fontsize = 20)
plt.show()

x = data['Annual Income (k$)']
y = data['Age']
z = data['Spending Score (1-100)']

sns.lineplot(x, y, color = 'blue')
sns.lineplot(x, z, color = 'pink')
plt.title('Annual Income vs Age and Spending Score', fontsize = 20)
plt.show()

############### Machine learning K-means Clustering #########
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans
n = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    n.append(kmeans.inertia_)
plt.plot(range(1, 11), n)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wss')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')
plt.title('Clusters_of_customers')
plt.xlabel('Annual_Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


df = pd.read_csv('Mall_Customers.csv', index_col=0)
#changing column names for better manipulability
df = df.rename(columns={'Genre': 'Genre', 'Age': 'age', 'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'})
df['Genre'].replace(['Female','Male'], [0,1],inplace=True)
df.head()
print(df.head())
#SCALING
#stocking mean and standard deviation in a dataframe (we will need them for unscaling)
dfsp = pd.concat([df.mean().to_frame(), df.std().to_frame()], axis=1).transpose()
dfsp.index = ['mean', 'std']
#new dataframe with scaled values
df_scaled = pd.DataFrame()
for c in df.columns:
    if(c=='Genre'): df_scaled[c] = df[c]
    else: df_scaled[c] = (df[c] - dfsp.loc['mean', c]) / dfsp.loc['std', c]
df_scaled.head()


#the two "intuitive" clusters
dff = df_scaled.loc[df_scaled.Genre==0].iloc[:, 1:] #no need of gender column anymore
dfm = df_scaled.loc[df_scaled.Genre==1].iloc[:, 1:]


def number_of_clusters(df):
    wcss = []
    for i in range(1,20):
        km=KMeans(n_clusters=i, random_state=0)
        km.fit(df)
        wcss.append(km.inertia_)

    df_elbow = pd.DataFrame(wcss)
    df_elbow = df_elbow.reset_index()
    df_elbow.columns= ['n_clusters', 'within_cluster_sum_of_square']

    return df_elbow

dfm_elbow = number_of_clusters(dfm)
dff_elbow = number_of_clusters(dff)

fig, ax = plt.subplots(1, 2, figsize=(17,5))

sns.lineplot(data=dff_elbow, x='n_clusters', y='within_cluster_sum_of_square', ax=ax[0])
sns.scatterplot(data=dff_elbow[5:6], x='n_clusters', y='within_cluster_sum_of_square', color='black', ax=ax[0])
ax[0].set(xticks=dff_elbow.index)
ax[0].set_title('Female')

sns.lineplot(data=dfm_elbow, x='n_clusters', y='within_cluster_sum_of_square', ax=ax[1])
sns.scatterplot(data=dfm_elbow[5:6], x='n_clusters', y='within_cluster_sum_of_square', color='black', ax=ax[1])
ax[1].set(xticks=dfm_elbow.index)
ax[1].set_title('Male');

def k_means(n_clusters, df, gender):

    kmf = KMeans(n_clusters=n_clusters, random_state=0) #defining the algorithm
    kmf.fit_predict(df) #fitting and predicting
    centroids = kmf.cluster_centers_ #extracting the clusters' centroids
    cdf = pd.DataFrame(centroids, columns=df.columns) #stocking in dataframe
    cdf['gender'] = gender
    return cdf

df1 = k_means(5, dff, 'female')
df2 = k_means(5, dfm, 'male')
dfc_scaled = pd.concat([df1, df2])
dfc_scaled.head()
print(dfc_scaled.head())

#UNSCALING
#using the mean and standard deviation of the original dataframe, stocked earlier
dfc = pd.DataFrame()
for c in dfc_scaled.columns:
    if(c=='gender'): dfc[c] = dfc_scaled[c]
    else: 
        dfc[c] = (dfc_scaled[c] * dfsp.loc['std', c] + dfsp.loc['mean', c])
        dfc[c] = dfc[c].astype(int)
        
dfc.head()
print(dfc.head())

################## Dendgram graph ###################################
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X , method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()