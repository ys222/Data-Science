library(caTools)
library(dplyr)
library(ggplot2)
library(caret)
library(class)
library(corrplot)

zoo <- read.csv("Zoo.csv")
View(zoo)
class(zoo)

summary(zoo)
str(zoo)

#Join the standardized data with the target column
data <- cbind(standard.features,zoo[10])
#Check if there are any missing values to impute. 
anyNA(data)

# Looks like the data is free from NA's
head(data)

corrplot(cor(data))

set.seed(101)

sample <- sample.split(data$Type,SplitRatio = 0.70)

train <- subset(data,sample==TRUE)

test <- subset(data,sample==FALSE)


predicted.type <- knn(train[1:9],test[1:9],train$Type,k=1)
#Error in prediction
error <- mean(predicted.type!=test$Type)


predicted.type <- NULL
error.rate <- NULL

for (i in 1:10) {
  predicted.type <- knn(train[1:9],test[1:9],train$Type,k=i)
  error.rate[i] <- mean(predicted.type!=test$Type)
  
}

knn.error <- as.data.frame(cbind(k=1:10,error.type =error.rate))

predicted.type <- knn(train[1:9],test[1:9],train$Type,k=3)
#Error in prediction
error <- mean(predicted.type!=test$Type)

