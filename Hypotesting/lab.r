# Read the data
data <- read.csv(file.choose())
View(data)
class(data)

#Author DataFlair
x = rnorm(10)
y = rnorm(10)
t.test(x,y)

t.test(x, y, var.equal = TRUE)
#Author DataFlair
t.test(x, mu = 5)

#Author DataFlair
t.test(y, mu = 5, alternative = 'greater')

#DataFlair
cor.test(data$Laboratory_1, data$Laboratory_2)

