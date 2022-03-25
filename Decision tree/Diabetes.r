library(corrplot)
library(caret)
pima <- read.csv("Diabetes.csv", col.names=c("Pregnant","Plasma_Glucose","Dias_BP","Triceps_Skin","Serum_Insulin","BMI","DPF","Age","Diabetes"))

head(pima)

str(pima)

sapply(pima, function(x) sum(is.na(x)))
pairs(pima, panel = panel.smooth)

corrplot(cor(pima[, -9]), type = "lower", method = "number")

# Preparing the DataSet
set.seed(123)
n <- nrow(pima)
train <- sample(n, trunc(0.70*n))
pima_training <- pima[train, ]
pima_testing <- pima[-train, ]

# Training The Model
glm_fm1 <- glm(Diabetes ~., data = pima_training, family = binomial)
summary(glm_fm1)

glm_fm2 <- update(glm_fm1, ~. - Triceps_Skin - Serum_Insulin - Age )
summary(glm_fm2)

par(mfrow = c(2,2))
plot(glm_fm2)

# Testing the Model
glm_probs <- predict(glm_fm2, newdata = pima_testing, type = "response")
glm_pred <- ifelse(glm_probs > 0.5, 1, 0)
#print("Confusion Matrix for logistic regression"); table(Predicted = glm_pred, Actual = pima_testing$Diabetes)
set.seed(1000)
intrain <- createDataPartition(y = pima$Diabetes, p = 0.7, list = FALSE)
train <- pima[intrain, ]
test <- pima[-intrain, ]

# Training The Model
treemod <- tree(Diabetes ~ ., data = train)

treemod
plot(treemod)
text(treemod, pretty = 0)

# Testing the Model
tree_pred <- predict(treemod, newdata = test, type = "class" )
confusionMatrix(tree_pred, test$Diabetes)

# Training The Model
set.seed(123)
library(randomForest)

rf_pima <- randomForest(Diabetes ~., data = pima_training, mtry = 8, ntree=50, importance = TRUE)

# Testing the Model
rf_probs <- predict(rf_pima, newdata = pima_testing)
rf_pred <- ifelse(rf_probs > 0.5, 1, 0)

importance(rf_pima)
par(mfrow = c(1, 2))
varImpPlot(rf_pima, type = 2, main = "Variable Importance",col = 'black')
plot(rf_pima, main = "Error vs no. of trees grown")

