## Importing packages



library(tidyverse) # metapackage with lots of helpful functions

## Running code

# In a notebook, you can run a single code cell by clicking in the cell and then hitting 
# the blue arrow to the left, or by clicking in the cell and pressing Shift+Enter. In a script, 
# you can run code by highlighting the code you want to run and then clicking the blue arrow
# at the bottom of this window.

## Reading in files

# You can access files from datasets you've added to this kernel in the "../input/" directory.
# You can see the files added to this kernel by running the code below. 

list.files(path = "../input")

## Saving data

# If you save any files or images, these will be put in the "output" directory. You 
# can see the output directory by committing and running your kernel (using the 
# Commit & Run button) and then checking out the compiled version of your kernel.

library(stats)
library(corrplot)
library(cluster)
library(factoextra)
library(ggplot2)
data("USArrests")
file1 <-as.data.frame(USArrests) 
colnames(file1)<-c("Murder","Assault","UrbanPop","Rape")
#Remove mising values
is.na(file1)
file1 <- na.omit(file1)

#Scaleing the variables
file1 <- scale(file1)

head(file1)

#Basic statistics
Min<-apply(USArrests, 2, min)
Max<-apply(USArrests, 2, max)
Med<-apply(USArrests, 2, median)
Sd<-apply(USArrests, 2, sd)
Mean<-apply(USArrests, 2, mean)
Bstat<<-data.frame(Min,Max,Med,Sd,Mean)

head(Bstat)

#correlation
f1<-scale(file1)
op1 <- dist(file1, method = "euclidean" )
print("Correlation method for distance measure")
cor <- cor(t(f1), method = "pearson")
dist_cor <- as.dist(1 - cor)
round(as.matrix(dist_cor)[1:6,1:6],1)


#visualization
corrplot(as.matrix(head(f1,15)), is.corr = FALSE, method = "ellipse",hclust = "complete")

corrplot(as.matrix(f1), is.corr = FALSE, method = "number", order="original", type = "upper")

plot(f1)
boxplot(f1)

#Another way for basic statistics
summary(file1)

data("USArrests")
mydata <- USArrests
#Remove mising values
mydata <- na.omit(mydata)

#Scale the variables
mydata <- scale(mydata)

head(mydata, n=10)

#Data preparation example

set.seed(124)
ss <- sample(1:50,10)
df <- USArrests[ss, ]
df <- na.omit(df)
head(df,n=6)

df.scaled <- scale(df)
head(round(df.scaled, 2))

#second argument take value where to apply the function 
#1. for function on rows
#2. for function on columns

desc_stats <- data.frame(
  Min = apply(USArrests, 2, min),
  Max = apply(USArrests, 2, max),
  Med = apply(USArrests, 2, median),
  SD = apply(USArrests, 2, sd),
  Mean = apply(USArrests, 2, mean)
)

desc_stats <- round(desc_stats,1)
head(desc_stats)

library(stats)
eucl <- dist(df.scaled, method = "euclidean" )
#method can be euclidean, manhattan, correlation etc
round(as.matrix(eucl)[1:6,1:6],1)
#t() used for transposing data

#correlation based distance#########################################
#Cor computes coefficient between variables but we need observations 
#so using t() to transpose matrix

print("Correlation method for distance measure")
cor <- cor(t(df.scaled), method = "pearson")
dist_cor <- as.dist(1 - cor)
round(as.matrix(dist_cor)[1:6,1:6],1)

#daisy() to compute dissimilarity matrices between observations 
library(cluster)
library(factoextra)
daisy(df.scaled, metric = c("euclidean", "manhattan", "gower"), stand = FALSE)

#stand: if TRUE, then the measurements in df.scaled are standardized before calculating the dissimilarities. Measurements are standardized for each variable (column), by subtracting the variable’s mean value and dividing by the variable’s mean absolute deviation

data("flower")
head(flower)

str(flower)

daisy_dist <- as.matrix(daisy(flower))
head(round(daisy_dist[1:6,1:6]),2)

#visualizing distance matrices

library(corrplot)
corrplot(as.matrix(eucl), is.corr = FALSE, method = "color")

corrplot(as.matrix(eucl), is.corr = FALSE, method = "color", order = "hclust", type = "upper")

#hierarchial clustering dendrogram 
plot(hclust(eucl, method = "ward.D2"))


#Heatmap
heatmap(as.matrix(eucl), symm = TRUE, distfun = function(x) as.dist(x))


library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra)

df <- USArrests

df <- na.omit(df)


df <- scale(df)
head(df)

distance <- get_dist(df)

k2 <- kmeans(df, centers = 2, nstart = 25)
str(k2)

k2

k3 <- kmeans(df, centers = 3, nstart = 25)
k4 <- kmeans(df, centers = 4, nstart = 25)
k5 <- kmeans(df, centers = 5, nstart = 25)

# Compute k-means clustering with k = 4
set.seed(123)
final <- kmeans(df, 4, nstart = 25)
print(final)

# Dissimilarity matrix
d <- dist(df, method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )

# Plot the obtained dendrogram
plot(hc1, cex = 0.6, hang = -1)




