# Libraries
library(naivebayes)
library(ggplot2)

library(caret)
library(psych)

library(e1071)

# Data(Train)
train_sal <- read.csv(file.choose())
str(train_sal)

View(train_sal)
train_sal$educationno <- as.factor(train_sal$educationno)
class(train_sal)

# Data(Test)
test_sal <- read.csv(file.choose())
str(test_sal)

View(test_sal)
test_sal$educationno <- as.factor(test_sal$educationno)
class(test_sal)

plot(train_sal$workclass,train_sal$Salary)

plot(train_sal$education,train_sal$Salary)

plot(train_sal$educationno,train_sal$Salary)

plot(train_sal$maritalstatus,train_sal$Salary)

plot(train_sal$occupation,train_sal$Salary)

plot(train_sal$relationship,train_sal$Salary)

plot(train_sal$race,train_sal$Salary)

plot(train_sal$sex,train_sal$Salary)

plot(train_sal$native,train_sal$Salary)

# Naive Bayes Model 
Model <- naiveBayes(train_sal$Salary ~ ., data = train_sal)
Model

Model_pred <- predict(Model,test_sal)
mean(Model_pred==test_sal$Salary)

confusionMatrix(Model_pred,test_sal$Salary)

boxplot(train_sal)
boxplot(test_sal)

