library(survival)
library(ranger)
library(ggplot2)
library(dplyr)
library(ggfortify)

# Data(Train)
patient <- read.csv(file.choose())
str(patient)

head(patient)

# Kaplan Meier Survival Curve
km <- with(patient, Surv(Followup, Eventtype))
head(km,80)

km_fit <- survfit(Surv(Followup, Eventtype) ~ 1, data=patient)
summary(km_fit, times = c(1,30,60,90*(1:10)))

#plot(km_fit, xlab="Days", main = 'Kaplan Meyer Plot') #base graphics is always ready
plot(km_fit)

km_fit <- survfit(Surv(Followup, Eventtype) ~ Scenario, data=patient)
plot(km_fit)

Pat <- mutate(patient, AG = ifelse((age < 60), "LT60", "OV60"),
              Followup = factor(AG),
              Scenario = factor(trt,labels=c("standard","test")),
              prior = factor(prior,labels=c("N0","Yes")))

km_fit <- survfit(Surv(Followup, Eventtype) ~ Followup, data=Pat)
plot(km_fit)

# Fit Cox Model
cox <- coxph(Surv(Followup, Eventtype)  , data = patient)
summary(cox)

# ranger model
r_fit <- ranger(Surv(Followup, Eventtype) ~ trt + celltype + 
                  karno + diagtime + age + prior,
                data = vet,
                mtry = 4,
                importance = "permutation",
                splitrule = "extratrees",
                verbose = TRUE)

# Average the survival models
death_times <- r_fit$unique.death.times 
surv_prob <- data.frame(r_fit$survival)
avg_prob <- sapply(surv_prob,mean)

# Plot the survival models for each patient
plot(r_fit$unique.death.times,r_fit$survival[1,], 
     type = "l", 
     ylim = c(0,1),
     col = "red",
     xlab = "Days",
     ylab = "survival",
     main = "Patient Survival Curves")

#
cols <- colors()
for (n in sample(c(2:dim(vet)[1]), 20)){
  lines(r_fit$unique.death.times, r_fit$survival[n,], type = "l", col = cols[n])
}
lines(death_times, avg_prob, lwd = 2)
legend(500, 0.7, legend = c('Average = black'))


vi <- data.frame(sort(round(r_fit$variable.importance, 4), decreasing = TRUE))
names(vi) <- "importance"
head(vi)

# Set up for ggplot
kmi <- rep("KM",length(km_fit$time))
km_df <- data.frame(km_fit$time,km_fit$surv,kmi)
names(km_df) <- c("Time","Surv","Model")

coxi <- rep("Cox",length(cox_fit$time))
cox_df <- data.frame(cox_fit$time,cox_fit$surv,coxi)
names(cox_df) <- c("Time","Surv","Model")

rfi <- rep("RF",length(r_fit$unique.death.times))
rf_df <- data.frame(r_fit$unique.death.times,avg_prob,rfi)
names(rf_df) <- c("Time","Surv","Model")

plot_df <- rbind(km_df,cox_df,rf_df)

p <- ggplot(plot_df, aes(x = Time, y = Surv, color = Model))
p + geom_line()



