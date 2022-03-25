library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra)
library(fpc)
library(NbClust)

# Reading the .csv file as a data frame
AirLine_DF = read.csv("AirlinesCluster.csv")

# Reading the structure of the data
str(AirLine_DF)

data = scale(AirLine_DF)

d <- dist(data[,2:4], method = "euclidean") 

fit <- hclust(d, method="ward.D2")
fit <- as.dendrogram(fit, k=3)

SampleSize = as.integer(0.95 * nrow(data))
sample1 = data[sample(1:nrow(data), SampleSize,replace=FALSE),]

dist.1 = dist(sample1[,2:11], method = "euclidean") 
fit.1 <- hclust(dist.1, method="ward.D2")
fit.1 <- as.dendrogram(fit.1)


sample2 = data[sample(1:nrow(data), SampleSize,replace=FALSE),]

dist.2 = dist(sample2[,2:11], method = "euclidean") 
fit.2 <- hclust(dist.2, method="ward.D2")
fit.2 <- as.dendrogram(fit.2)


        
