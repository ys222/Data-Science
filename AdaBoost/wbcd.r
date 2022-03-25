library("FRESA.CAD")
library("mlbench")
library(fastAdaboost)
library(gbm)


# Read the data
data <- read.csv(file.choose())
View(data)
class(data)

data<- data[complete.cases(data),]
data$id <- 1*(data$id == "pos")


ExperimentName <- "data"
theData <- data.mat;
theOutcome <- "radius_mean";
reps <- 60;
fraction <- 0.5;

CVFileName <- paste(ExperimentName,"CVMethod_v2.RDATA",sep = "_")
op <- par(no.readonly = TRUE)

GBM_fit <- function(formula = formula, data=NULL,distribution = "bernoulli",n.trees = 1000,
                    shrinkage = 0.01, interaction.depth = 4,...)
  
{
  fit <- gbm(formula=formula,data = data,distribution = distribution,n.trees = n.trees,
             shrinkage = shrinkage, interaction.depth = interaction.depth,...);
  selectedfeatures <- summary(fit,plotit = FALSE);
  sum <- 0;
  sfeat = 1;
  while (sum < 90) {sum <- sum + selectedfeatures[sfeat,2]; sfeat <- sfeat + 1;} #keep the ones that add to 90%
  
  result <- list(fit=fit,n.trees=n.trees,selectedfeatures=rownames(selectedfeatures[1:sfeat,]))
  class(result) <- "GBM_FIT";
  return(result)
}  
  
  
  
  