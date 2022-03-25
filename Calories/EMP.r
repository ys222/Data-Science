Emp_data <- read.csv("emp_data.csv") 

summary(Emp_data)

# Variance and Standard deviation of Salary_hike column
var(Emp_data$Salary_hike)

sd(Emp_data$Salary_hike)

var(Emp_data$Churn_out_rate)

sd(Emp_data$Churn_out_rate)

Churn_out_rate_Model <- lm(Churn_out_rate ~ Salary_hike, data = Emp_data)
summary(Churn_out_rate_Model)

plot(Churn_out_rate_Model)

