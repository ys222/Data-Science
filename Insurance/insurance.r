library(data.table)



Insurance_Dataset <- read.csv("Insurance Dataset.csv")

summary(Insurance_Dataset)

Norm_Insrance <- scale(Insurance_Dataset)


library(cluster)

insurance_kmeans <- kmeans(Norm_Insrance,3)


insurance_clara <- clara(Norm_Insrance,3)

Insurance_Dataset_clara <- cbind(Insurance_Dataset,insurance_clara$cluster)

clusplot(insurance_clara)

library(animation)

km1 <- kmeans.ani(Norm_Insrance,3)

insurance_pam <- pam(Norm_Insrance,3)

Insurance_Dataset_pam <- cbind(Insurance_Dataset,insurance_pam$cluster)

clusplot(insurance_pam)

# Hierarichal CLustering
dist_insurance <- dist(Norm_Insrance, method = "euclidean")



hclust_ins <- hclust(dist_insurance, method = "complete")



plot(hclust_ins, hang = -1)



rect.hclust(hclust_ins,plot(hclust_ins,hang=-1),k=3,border="red")

group_ins <- cutree(hclust_ins, k=3)



Insurance_Dataset_Final <- cbind(Insurance_Dataset,group_ins)



aggregate(Insurance_Dataset_Final, by= list(Insurance_Dataset_Final$group_ins), FUN = mean)

