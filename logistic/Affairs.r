library('AER')
library(plyr)

# Read the data
Affairs <- read.csv(file.choose())
View(Affairs)
class(Affairs)

affairs1 <- Affairs
summary(affairs1)

table(affairs1$affairs)

affairs1$ynaffairs[affairs1$affairs > 0] <- 1
affairs1$ynaffairs[affairs1$affairs == 0] <- 0
affairs1$gender <- as.factor(revalue(Affairs$gender,c("male"=1, "female"=0)))
affairs1$children <- as.factor(revalue(Affairs$children,c("yes"=1, "no"=0)))
# sum(is.na(claimants))
# claimants <- na.omit(claimants) # Omitting NA values from the Data 
# na.omit => will omit the rows which has atleast 1 NA value
View(affairs1)


colnames(affairs1)

class(affairs1)

attach(affairs1)

# Preparing a linear regression 
mod_lm <- lm(naffairs ~ factor(unhap) + unhap+ yrsmarr1+ factor(kids) + vryhap+
               vryrel+vryunhap+avgmarr, data = affairs1)
summary(mod_lm)

pred1 <- predict(mod_lm,affairs1)
pred1

# plot(affairs,pred1)
# We can no way use the linear regression technique to classify the data
plot(pred1)

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(naffairs ~ factor(unhap) + unhap+ yrsmarr2+ factor(kids) + vryhap+
               vryrel+vryunhap+avgmarr, data = affairs1)

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Confusion matrix table 
prob <- predict(model,affairs1,type="response")
summary(model)

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
yes_no <- NULL

pred_values <- ifelse(prob>=0.5,1,0)
yes_no <- ifelse(prob>=0.5,"yes","no")

# Creating new column to store the above values
affairs1[,"prob"] <- prob
affairs1[,"pred_values"] <- pred_values
affairs1[,"yes_no"] <- yes_no

View(affairs1[,c(1,9:11)])

table(affairs1$ynaffairs,affairs1$pred_values)

