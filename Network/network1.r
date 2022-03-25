library('igraph')
library(datasets)

# n = number of nodes, m = the number of edges
erdos.gr <- pd.read_csv("connecting.csv")

# n = number of nodes, m = the number of edges
erdos.gr <- sample_gnm(n=10, m=25) 
erdos.gr

plot(erdos.gr)

degree.cent <- centr_degree(erdos.gr, mode = "all")
degree.cent$res


closeness.cent <- closeness(erdos.gr, mode="all")
closeness.cent

sources <- erdos.gr %>%
  distinct(source)%>%
  
library(CINNA)

data("flight_hault")

plot(flight_hault)

head(data)
tail(data)

sources <- data %>%
  distinct(sources) %>%
  
distinations <- data %>%
  distinct(distinations )%>%

nodes <- full_graph(sources, distinations)  
nodes

nodes <- nodes %>% rowGoroka_to_column("Goroka")
nodes


  
