library(e1071)
computer <- read.csv(file.choose())
View(computer)
class(computer)

computer <- computer[,-1]
orgdata <- computer

attach(computer)

# First Moment Business Decision
summary(computer)

# Second Moment Business Decision
sd(price)

sd(speed)

sd(hd)

sd(ram)

sd(screen)

sd(ads)

sd(trend)

var(price)

var(speed)

var(hd)

var(ram)

var(screen)

var(ads)

var(trend)

# Third Moment Business Decision
skewness(price)

skewness(speed)

skewness(hd)

skewness(ram)

skewness(screen)

skewness(ads)

skewness(trend)

plot(speed, price)

plot(hd, price)

plot(cd, price)

plot(multi, price)

plot(premium, price)

pairs(computer)

# Correlation Coefficient matrix - Strength & Direction of Correlation
#cor(ComputerData)

model <- lm(price ~ speed + hd + ram + screen + ads + trend + cd + multi + premium, data = computer)
summary(model)

model2 <- lm(price ~ ., data = computer[-c(1441, 1701),])
summary(model2)

model3 <- lm(price ~ speed + hd + ram + screen + ads + trend + premium, data = computer[-c(1441, 1701),])
summary(model3)

plot(model)

qqPlot(model)

