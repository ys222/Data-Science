data<-read.csv('bank_data.csv',sep = ';')
head(data)

str(data)

datamodel<-glm(data = data)
summary(datamodel)

data$predict<-predict(datamodel,data=data,type='response')
cm=table(data$y,data$predict>0.5)
cm

n = sum(cm) # number of instances
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
accuracy = sum(diag) / n 
accuracy

precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1)