library(C50)
library(tree)
library(gmodels)
library(party)
library(caret)

CompanyData <- read.csv(file.choose())
# Splitting data into training and testing.
# splitting the data based on Sales
hist(CompanyData$Sales)


High = ifelse(CompanyData$Sales<10, "No", "Yes")
CD = data.frame(CompanyData, High)
#CD <- CompanyData[,2:12]
# View(CD)

CD_train <- CD[1:200,]

# View(CD_train)
CD_test <- CD[201:400,]

# View(CD_test)

#Using Party Function 
op_tree = ctree(High ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc
                + Age + Education + Urban + US, data = CD_train)
summary(op_tree)


plot(op_tree)

# On looking into the Above tree, i see that if the Location of the Shelv is good,
# then there is a probability of 60% chance that the customer will buy.
# With ShelveLoc having a Bad or Medium and Price <= 87, the probability of High sales 
# could be 60%.
# If ShelveLoc is Bad or Medium, With Price >= 87 and Advertising less then <= 7 then there
# is a zero percent chance of high sales.
# If ShelveLoc is Bad or Medium, With Price >= 87 and Advertising less then > 7 then there
# is a 20 % percent chance of high sales.
pred_tree <- as.data.frame(predict(op_tree,newdata=CD_test))
pred_tree["final"] <- NULL
pred_test_df <- predict(op_tree,newdata=CD_test)


mean(pred_test_df==CD$High)


CrossTable(CD_test$High,pred_test_df)

confusionMatrix(CD_test$High,pred_test_df)

##### Using tree function 
cd_tree_org <- tree(High~.-Sales,data=CD)
summary(cd_tree_org)


plot(cd_tree_org)
text(cd_tree_org,pretty = 0)

# Using the training data

##### Using tree function 
cd_tree <- tree(High~.-Sales,data=CD_train)
summary(cd_tree)

plot(cd_tree)
text(cd_tree,pretty = 0)

### Evaluate the Model

# Predicting the test data using the model
pred_tree <- as.data.frame(predict(cd_tree,newdata=CD_test))
pred_tree["final"] <- NULL
pred_test_df <- predict(cd_tree,newdata=CD_test)


pred_tree$final <- colnames(pred_test_df)[apply(pred_test_df,1,which.max)]

pred_tree$final <- as.factor(pred_tree$final)
summary(pred_tree$final)


summary(CD_test$High)

mean(pred_tree$final==CD$High) 

CrossTable(CD_test$High,pred_tree$final)


