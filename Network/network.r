library('igraph')
library(datasets)

# n = number of nodes, m = the number of edges
erdos.gr <- choose.files()
View(erdos.gr)

# n = number of nodes, m = the number of edges
erdos.gr <- sample_gnm(n=10, m=25) 

plot(erdos.gr)

degree.cent <- centr_degree(erdos.gr, mode = "all")
degree.cent$res


closeness.cent <- closeness(erdos.gr, mode="all")
closeness.cent

library(CINNA)

data("linkedin")

plot(linkedin)

pr_cent<-proper_centralities(linkedin)

calculate_centralities(linkedin, include = pr_cent[1:5])%>%
pca_centralities(scale.unit = TRUE)

visualize_graph( linkedin , centrality.type="Barycenter Centrality")

