library("recommenderlab")  

# Loading to pre-computed affinity data
mydata<- (file.choose())
affinity.data<- read_excel("Ratings.xlsx")
affinity.matrix<- as(affinity.data,"realRatingMatrix")

# Creation of the model - U(ser) B(ased) C(ollaborative) F(iltering)
Rec.model<-Recommender(affinity.matrix[1:5000], method = "UBCF")


eval_recommender = Recommender(data = getData(eval_sets, "train"),
                               method = "UBCF", parameter = NULL)
items_to_recommend = 10
eval_prediction = predict(object = eval_recommender,
                          newdata = getData(eval_sets, "known"),
                          n = items_to_recommend,
                          type = "ratings")
eval_accuracy = calcPredictionAccuracy(x = eval_prediction,
                                       data = getData(eval_sets, "unknown"),
                                       byUser = TRUE)
head(eval_accuracy)


# Draw ROC curve
plot(results, y = "ROC", annotate = 1, legend="topleft")
title("ROC Curve")


# Draw precision / recall curve
plot(results, y = "prec/rec", annotate=1)
title("Precision-Recall")