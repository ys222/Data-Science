Startups <- read.csv(file.choose())
View(Startups)
class(Startups)

# To Transform the data from Character to Numeric
library(plyr)

Startups$State <- revalue(Startups$State,
                          c("New York"="0", "California"="1", "Florida"="2")) 
attach(Startups)
Startups <- cbind(RD_Spend=R.D.Spend,Administration,Marketing_Spend=Marketing.Spend,State,Profit)


Startups <- as.data.frame(Startups)

attach(Startups)

summary(Startups)

plot(R.D.Spend, Profit)

plot(Administration, Profit)

plot(Marketing.Spend, Profit)

plot(State, Profit)

windows()
# 7. Find the correlation between Output (Profit) & inputs (R.D Spend, Administration, Marketing, State) - SCATTER DIAGRAM
pairs(Startups)

# 8. Correlation coefficient - Strength & Direction of correlation
cor(Startups)

# The Linear Model of interest
Model.Startups <- lm(Profit~RD_Spend+Administration+Marketing_Spend+State)
summary(Model.Startups)

Model.Startups1 <- lm(Profit~RD_Spend+log(Administration))
summary(Model.Startups1)

### Scatter plot matrix with Correlations inserted in graph
panel.cor <- function(x, y, digits=2, prefix="", cex.cor)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r = (cor(x, y))
  txt <- format(c(r, 0.123456789), digits=digits)[1]
  txt <- paste(prefix, txt, sep="")
  
  if(missing(cex.cor)) cex <- 0.4/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex)
}
pairs(Startups, upper.panel=panel.cor,main="Scatter Plot Matrix with Correlation Coefficients")

library(corpcor)
cor2pcor(cor(Startups))

# install.packages("mvinfluence")
library(mvinfluence)

library(car)

# It is better to delete a single observation rather than entire variable to get rid of collinearity problem
# Deletion Diagnostics for identifying influential variable
influence.measures(Model.Startups)

# Logarthimic Transformation 
Model.Startups_Log<-lm(Profit~RD_Spend+log(Administration)+Marketing_Spend+log(State),data=Startups[-c(49,50),]) 

summary(Model.Startups_Log)

confint(Model.Startups_Log,level=0.95)

predict(Model.Startups_Log,interval="predict")

Model.Startups_Fin1<-lm(Profit~RD_Spend+Administration+Marketing_Spend+State,data=Startups[-c(49,50),])
summary(Model.Startups_Fin1) 


# Exponential Transformation :
Model.Startups_exp<-lm(log(Profit)~RD_Spend+Administration+Marketing_Spend+State,data=Startups[-c(49,50),])
summary(Model.Startups_exp)

Model.Startups_exp1<-lm(log(Profit)~RD_Spend+Marketing_Spend,data=Startups[-c(49,50),])
summary(Model.Startups_exp1) 

# Quad Model
Model.Startups_Quad <- lm(Profit~RD_Spend+I(RD_Spend^2)+Administration+I(Administration^2)
                          +Marketing_Spend+I(Marketing_Spend^2)+State+I(State^2),data=Startups[-c(49,50),])
summary(Model.Startups_Quad) 

confint(Model.Startups_Quad,level=0.95)

predict(Model.Startups_Quad,interval="predict")

Model.Startups_Quad1 <- lm(Profit~RD_Spend+I(RD_Spend^2)+Marketing_Spend+I(Marketing_Spend^2)
                           ,data=Startups[-c(49,50),])
summary(Model.Startups_Quad1)

# Poly Modal
Model.Startups_Poly <- lm(Profit~RD_Spend+I(RD_Spend^2)+I(RD_Spend^3)+
                            Administration+I(Administration^2)+I(Administration^3)+
                            Marketing_Spend+I(Marketing_Spend^2)+I(Marketing_Spend^3)+
                            State+I(State^2)+I(State^3),data=Startups[-c(49,50),])
summary(Model.Startups_Poly)

Model.Startups_Poly1 <- lm(Profit~RD_Spend+I(RD_Spend^2)+I(RD_Spend^3)+
                             Marketing_Spend+I(Marketing_Spend^2)+I(Marketing_Spend^3)
                           ,data=Startups[-c(49,50),])
summary(Model.Startups_Poly1) 

# Final Model
FinalModel<-lm(Profit~RD_Spend+log(Administration)+Marketing_Spend+
                 log(State),data=Startups[-c(49,50),])

summary(FinalModel)

Profit_Predict <- predict(FinalModel,interval="predict")

Final <- cbind(Startups$RD_Spend,Startups$Administration,Startups$Marketing_Spend,
               Startups$State,Startups$Profit,Profit_Predict)

View(Final)


# Evaluate model LINE assumptions
plot(FinalModel)

qqPlot(FinalModel, id.n=5)


