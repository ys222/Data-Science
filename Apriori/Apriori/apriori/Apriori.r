library(tidyverse) # helpful in Data Cleaning and Manipulation
library(arules) # Mining Association Rules and Frequent Itemsets
library(arulesViz) # Visualizing Association Rules and Frequent Itemsets 
library(gridExtra) #  low-level functions to create graphical objects 
library(ggthemes) # For cool themes like fivethirtyEight
library(dplyr) # Data Manipulation
library(readxl)# Read Excel Files in R
library(plyr)# Tools for Splitting, Applying and Combining Data
library(ggplot2) # Create graphics and charts
library(knitr) # Dynamic Report generation in R
library(lubridate) # Easier to work with dates and times.
library(kableExtra) # construct complex tables and customize styles
library(RColorBrewer) # Color schemes for plotting

data <- read.csv("book.csv")

head(data)

tail(data)

str(data)

Norm_Apriori <- scale(data)

rules <- apriori(data,
                 
                 parameter = list(supp = 0.1, conf = 0.100))

cbind(data$ChildBks)
cbind(data$YouthBks)
cbind(data$CookBks)
cbind(data$DoItYBks)


library(plyr)
#ddply(dataframe, variables_to_be_used_to_split_data_frame, function_to_be_applied)
data <- ddply(data,c("ChildBks","YouthBks"),
                         function(df1)paste(data$CookBks,
                                            collapse = ","))

hist(data$ChildBks)


df <- read.csv("groceries.csv")

library(arules)
class(df)

frequentItems <- eclat (df, parameter = list(supp = 0.07, maxlen = 15))

rules <- apriori (df, parameter = list(supp = 0.001, conf = 0.5))

rules_conf <- sort (rules, by="confidence", decreasing=TRUE)

rules_lift <- sort (rules, by="lift", decreasing=TRUE)

rules <- apriori (data=df, parameter=list (supp=0.001,conf = 0.08)) 

#Apriori algorithm Implementation below

# First, get itemsets of length 1
itemsets<-apriori(df,parameter=list(minlen=1,maxlen=1,support=0.02,target="frequent itemsets"))

summary(itemsets) 

# Second, get itemsets of length 2
itemsets<-apriori(df,parameter=list(minlen=2,maxlen=2,support=0.02,target="frequent itemsets"))
summary(itemsets)

# The Apriori function  () is used to generate rules. A threshold is set lower than 0.001 and minimum confidence threshold is set to 0.6. Below code generates 2,918 rules.
rules <- apriori(df,parameter=list(support=0.001,confidence=0.6,target="rules"))

summary(rules)

# Compute the 1/Support(Y) ie slope
slope<-sort(round(rules@quality$lift/rules@quality$confidence,2))

#Display the number of times each slope appears in dataset
unlist(lapply(split(slope,f=slope),length))

#Below code fetchces rules with confidence above 0.9
confidentRules<-rules[quality(rules)$confidence>0.9] 
confidentRules 

#Visualize the top 5 rules with the highest lift and plot them
highLiftRules<-head(sort(rules,by="lift"),5) 

df1 <- read.csv("my_movies.csv")

rules <- apriori(as.matrix(df1[,6:15],parameter=list(support=0.2, confidence = 0.5,minlen=5)))

rules

head(quality(rules))

rules_conf <- sort (rules, by="confidence", decreasing=TRUE)
rules_lift <- sort (rules, by="lift", decreasing=TRUE)

# First, get itemsets of length 1
itemsets<-apriori(df1,parameter=list(minlen=1,maxlen=1,support=0.02,target="frequent itemsets"))

summary(itemsets)


# Second, get itemsets of length 2
itemsets<-apriori(df1,parameter=list(minlen=2,maxlen=2,support=0.02,target="frequent itemsets"))
summary(itemsets)

# The Apriori function  () is used to generate rules. A threshold is set lower than 0.001 and minimum confidence threshold is set to 0.6. Below code generates 2,918 rules.
rules <- apriori(df1,parameter=list(support=0.001,confidence=0.6,target="rules"))

summary(rules)

# Compute the 1/Support(Y) ie slope
slope<-sort(round(rules@quality$lift/rules@quality$confidence,2))

#Display the number of times each slope appears in dataset
unlist(lapply(split(slope,f=slope),length))

#Below code fetchces rules with confidence above 0.9
confidentRules<-rules[quality(rules)$confidence>0.9] 
confidentRules 

#Visualize the top 5 rules with the highest lift and plot them
highLiftRules<-head(sort(rules,by="lift"),5) 

##for my phone dataset
df2 <-  read.csv("myphonedata.csv")


rules <- apriori(as.matrix(df2[,4:5],parameter=list(support=0.2, confidence = 0.5,minlen=5)))

rules

head(quality(rules))

rules_conf <- sort (rules, by="confidence", decreasing=TRUE)
rules_lift <- sort (rules, by="lift", decreasing=TRUE)

# First, get itemsets of length 1
itemsets<-apriori(df2,parameter=list(minlen=1,maxlen=1,support=0.02,target="frequent itemsets"))

summary(itemsets)


# Second, get itemsets of length 2
itemsets<-apriori(df2,parameter=list(minlen=2,maxlen=2,support=0.02,target="frequent itemsets"))
summary(itemsets)

# The Apriori function  () is used to generate rules. A threshold is set lower than 0.001 and minimum confidence threshold is set to 0.6. Below code generates 2,918 rules.
rules <- apriori(df2,parameter=list(support=0.001,confidence=0.6,target="rules"))

summary(rules)

# Compute the 1/Support(Y) ie slope
slope<-sort(round(rules@quality$lift/rules@quality$confidence,2))

#Display the number of times each slope appears in dataset
unlist(lapply(split(slope,f=slope),length))

#Below code fetchces rules with confidence above 0.9
confidentRules<-rules[quality(rules)$confidence>0.9] 
confidentRules 

#Visualize the top 5 rules with the highest lift and plot them
highLiftRules<-head(sort(rules,by="lift"),5) 

##for transactions_retail dataset
df3 <- read.csv("transactions_retail1.csv")

rules <- apriori(as.matrix(df3[,4:5],parameter=list(support=0.2, confidence = 0.5,minlen=5)))

rules

head(quality(rules))

rules_conf <- sort (rules, by="confidence", decreasing=TRUE)
rules_lift <- sort (rules, by="lift", decreasing=TRUE)

# First, get itemsets of length 1
itemsets<-apriori(df3,parameter=list(minlen=1,maxlen=1,support=0.02,target="frequent itemsets"))

summary(itemsets)


# Second, get itemsets of length 2
itemsets<-apriori(df3,parameter=list(minlen=2,maxlen=2,support=0.02,target="frequent itemsets"))
summary(itemsets)

# The Apriori function  () is used to generate rules. A threshold is set lower than 0.001 and minimum confidence threshold is set to 0.6. Below code generates 2,918 rules.
rules <- apriori(df3,parameter=list(support=0.001,confidence=0.6,target="rules"))

summary(rules)

# Compute the 1/Support(Y) ie slope
slope<-sort(round(rules@quality$lift/rules@quality$confidence,2))

#Display the number of times each slope appears in dataset
unlist(lapply(split(slope,f=slope),length))

#Below code fetchces rules with confidence above 0.9
confidentRules<-rules[quality(rules)$confidence>0.9] 
confidentRules 

#Visualize the top 5 rules with the highest lift and plot them
highLiftRules<-head(sort(rules,by="lift"),5) 




