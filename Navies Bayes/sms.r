sms = read.csv("sms_raw_NB.csv", stringsAsFactors=F)
str(sms)

round(prop.table(table(sms$type))*100, digits = 1)

sms$type = factor(sms$type)
#install.packages("tm")
library(tm)

# split the raw data:
sms.train = sms[1:4200, ] # about 75%
sms.test  = sms[4201:5574, ] # the rest

# let's just assert that our split is reasonable: raw data should have about 87% ham
# in both training and test sets:
round(prop.table(table(sms.train$type))*100)

round(prop.table(table(sms.test$type))*100)

#install.packages("wordcloud")
library(wordcloud)
convert_counts = function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels=c("No", "Yes"))
  return (x)
}

#install.packages("e1071")
library(e1071)
# store our model in sms_classifier
sms_classifier = naiveBayes(reduced_dtm.train, sms.train$type)
sms_test.predicted = predict(sms_classifier,
                             reduced_dtm.test)

# once again we'll use CrossTable() from gmodels
#install.packages("gmodels")
library(gmodels)
CrossTable(sms_test.predicted,
           sms.test$type,
           prop.chisq = FALSE, # as before
           prop.t     = FALSE, # eliminate cell proprtions
           dnn        = c("predicted", "actual")) # relabels rows+cols

sms_classifier2 = naiveBayes(reduced_dtm.train,
                             sms.train$type,
                             laplace = 1)
sms_test.predicted2 = predict(sms_classifier2,
                              reduced_dtm.test)
CrossTable(sms_test.predicted2,
           sms.test$type,
           prop.chisq = FALSE, # as before
           prop.t     = FALSE, # eliminate cell proprtions
           dnn        = c("predicted", "actual")) # relabels rows+cols



