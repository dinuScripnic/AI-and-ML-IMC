#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Working Directory
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Libraries
library(data.table)
library(plotly)
library(cluster) # additional clustering algorithms
library(factoextra) # A lot of nice visualizations and eval for clustering
library(dendextend)
library(GGally)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MALL dataset ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dt <- fread("./Lecture_08_Mall_Customers.csv")
str(dt)
dt[, Gender := as.factor(Gender)]
dt[, CustomerID := NULL]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## EDA ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# NA?
colSums(is.na(dt))

# Distribution per gender
summary(dt)
plotly::subplot(
  plot_ly(data = dt, 
          x = ~Gender,
          y = ~`Annual Income (k$)`,
          type = "box", 
          boxmean = T, 
          boxpoints = "all"),
  plot_ly(data = dt, 
          x = ~Gender, 
          y = ~`Spending Score (1-100)`,
          type = "box", 
          boxmean = T, 
          boxpoints = "all",
          showlegend = F),
  plot_ly(data = dt, 
          x = ~Gender,
          y = ~`Age`, 
          type = "box",
          boxmean = T,
          boxpoints = "all",
          showlegend = F), 
  titleY = T, 
  margin = 0.05)


# Density plot
density <- list("Age" = density(dt[,Age]), 
                "Income" = density(dt[,`Annual Income (k$)`]),
                "SS" = density(dt[,`Spending Score (1-100)`]))
plot_ly() |>
  add_trace(x = density[["Age"]]$x, 
            y = density[["Age"]]$y, 
            type = "scatter", 
            mode = "lines", 
            fill = "tozeroy", 
            name = "Age") |>
  add_trace(x = density[["Income"]]$x, 
            y = density[["Income"]]$y, 
            type = "scatter", 
            mode = "lines", 
            fill = "tozeroy", 
            name = "Income") |>
  add_trace(x = density[["SS"]]$x,
            y = density[["SS"]]$y,
            type = "scatter",
            mode = "lines",
            fill = "tozeroy", 
            name ="SS")

# bivariate analysis
#GGally::ggpairs(dt)
plot_ly(dt,
        type = "splom", 
        dimensions = list(
          list(label = "Income", values = ~`Annual Income (k$)`),
          list(label = "SS", values = ~`Spending Score (1-100)`),
          list(label = "Age", values = ~Age))) |>
  style(showupperhalf = F, diagonal = list(visible = F))

plot_ly(dt, 
        x = ~`Annual Income (k$)`,
        y = ~`Spending Score (1-100)`,
        type = "scatter", 
        mode = "markers",
        color = ~Gender,
        colors = c("Red","Blue"))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Hierarchical clustering ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We try clustering by using only spending score and Annual income
dt_new <- dt[, .(`Spending Score (1-100)`, `Annual Income (k$)`)]

### Distance ----
# Compute the distance
dist <- get_dist(dt_new, "euclidean")

# we also try with rescaled (center around the mean and rescale by using stdev)
dt_rescaled <- scale(dt_new)
dist_rescaled <- get_dist(dt_rescaled, "euclidean")

# Visualize the distance (in a sorted heatmap)
fviz_dist(dist_rescaled)
fviz_dist(dist) 

# AGGLOMERATIVE: hclust() 
# AGGLOMERATIVE: agnes() 
# DIVISIVE:      diana() 

hcl <- list()
for (linkage in c("average", "single", "complete", "ward")) {
  hcl[[linkage]] <- agnes(dist_rescaled, method = linkage)
}
#View(hcl)
sapply(hcl, \(x) x$ac)
lapply(hcl, \(x) plot(x, which.plots = 2, main=paste0("Dendrogram for ", x$method, " linkage")))

### Dendrogram ----
plot(hcl[["ward"]], which.plots=1) # Bannerplot
plot(hcl[["ward"]], which.plots=2) # Dendrogram
abline(h=5)
fviz_dend(hcl[["ward"]], k=5) # k=Num of clusters

tanglegram(
  dendlist(as.dendrogram(hcl[["ward"]]) |> set("labels_col", k=5) |> set("branches_k_color", k = 5),
           as.dendrogram(hcl[["ward"]]) |> set("labels_col", k=6) |> set("branches_k_color", k = 6)),
  common_subtrees_color_lines = FALSE, 
  highlight_distinct_edges  = TRUE, 
  highlight_branches_lwd=FALSE)

### Cut the tree ----
hcl_cut5 <- cutree(hcl[["ward"]], k=5)
hcl_cut6 <- cutree(hcl[["ward"]], k=6)


### Analysis ----

# VISUALIZE THE CLUSTERS
fviz_cluster(list(data=dt_new,cluster=hcl_cut5)) # if there are more than 2 dim, it uses PCA
fviz_cluster(list(data=dt_new,cluster=hcl_cut6)) # if there are more than 2 dim, it uses PCA
# Here we see with all possible 2 dimensions
# The following can probably be done with facets. I 
combinations <- combn(colnames(dt[,-1]), 2)
for(i in 1:ncol(combinations))
  print(fviz_cluster(list(data=dt[,-1],cluster=hcl_cut5), choose.vars = combinations[,i]))

# VISUALIZE THE SILHOUETTE
fviz_silhouette(silhouette(hcl_cut5, dist_rescaled))
fviz_silhouette(silhouette(hcl_cut6, dist_rescaled))

### Final Clusters ----
dt[,cluster:=hcl_cut5]
dt[,lapply(.SD,mean), cluster, .SDcols=is.integer]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## K-Means ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dt[,cluster:=NULL]
### Choose K ----
cls <- data.table(k = seq(2, 20, 1), WSS = 0, SS = 0)
for (i in cls[, k]) {
  cl <- kmeans(dt_rescaled, centers = i, nstart = 50)
  wss <- cl$tot.withinss
  ss <- ifelse(i != 1, mean(silhouette(cl$cluster, dist(dt))[, 3]), 0)
  cls[k == i, ':='(WSS = wss, SS = ss)]
}
plot_ly(data = cls, type = "scatter", mode = "lines") |>
  add_trace(x = ~k, y = ~WSS, name = "WSS") |>
  add_trace(x = ~k, y = ~SS, yaxis = "y2", name = "Silhouette Score", line = list(dash = "dash")) |>
  layout(yaxis2 = list(overlaying = "y", side = "right"))

### Final Cluster ----
# Elbow at 5
km <- kmeans(dt_rescaled, centers = 5, nstart = 25)
fviz_cluster(km, data=dt_rescaled, cluster=km)
combinations <- combn(colnames(dt[,-1]), 2)
for(i in 1:ncol(combinations))
  print(fviz_cluster(data=dt[,-1], object=km, choose.vars = combinations[,i]))

# What about age?
dt_new <- dt[, .(Age,`Annual Income (k$)`,`Spending Score (1-100)`)]
dt_rescaled<-scale(dt_new)
dist_rescaled<-dist(dt_rescaled,method="euclidean")
hcl_with_age_resc <- agnes(dt_rescaled, method = "ward")
fviz_dend(hcl_with_age_resc)
hcl_cut6 <- cutree(hcl_with_age_resc, k=6)
fviz_cluster(list(data=dt,cluster=hcl_cut6)) # if there are more than 2 dim, it uses PCA
combinations <- combn(colnames(dt), 2)
for(i in 1:ncol(combinations))
  print(fviz_cluster(list(data=dt, cluster=hcl_cut6), choose.vars = combinations[,i]))
fviz_silhouette(silhouette(hcl_cut6, dist_rescaled))

# What about the gender?
dt[,Gender:=ifelse(Gender=="Male",1,0)]
setnames(dt, "Gender", "isMale")
dt

# with rescaling
dt_rescaled <- scale(dt)
dist_rescaled <- dist(dt_rescaled, method = "euclidean")
hcl_with_gen_resc <- agnes(dt_rescaled, method = "ward")
fviz_dend(hcl_with_gen_resc)
hcl_with_gen_resc_10 <- cutree(hcl_with_gen_resc, k = 10)
fviz_cluster(list(data = dt, cluster = hcl_with_gen_resc_10)) # if there are more than 2 dim, it uses PCA
combinations <- combn(colnames(dt), 2)
for(i in 1:ncol(combinations))
  print(fviz_cluster(list(data = dt, cluster = hcl_with_gen_resc_10), choose.vars = combinations[,i]))
fviz_silhouette(silhouette(hcl_with_gen_resc_10, dist_rescaled))

# without rescaling
dist <- dist(dt, method = "euclidean")
hcl_with_gen_no_resc <- agnes(dist, method = "ward")
fviz_dend(hcl_with_gen_no_resc)
hcl_with_gen_no_resc_6 <- cutree(hcl_with_gen_no_resc, k = 6)
fviz_cluster(list(data = dt, cluster = hcl_with_gen_no_resc_6)) # if there are more than 2 dim, it uses PCA
combinations <- combn(colnames(dt), 2)
for(i in 1:ncol(combinations))
  print(fviz_cluster(list(data = dt, cluster = hcl_with_gen_no_resc_6), choose.vars = combinations[,i]))

fviz_silhouette(silhouette(hcl_with_gen_no_resc_6, dist))

# CONCLUSION: 
# IT IS NOT ADVISED TO USE FEATURES OF DIFFERENT TYPE IN THE EUCLIDEAN DISTANCE,
# WITHOUT TAKING PROPER PREPROCESSING STEPS (OR USE OTHER DISTANCES, E.G. GOWER).

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IRIS DATASET ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Data import ----
dt <- iris
setDT(dt)
# Let's shuffle
set.seed(123)
dt<-dt[sample(1:nrow(dt))]

## Preprocessing ----
# Let's assume we do not know the labels. 
dt[, .N, Species]
labels<-dt[, Species]
dt[, Species := NULL]

# Rescale (center around the mean and rescale by using stdev)
dt_rescaled <- scale(dt)

## Hierarchical clustering ----
dist <- get_dist(dt, "euclidean")
dist_rescaled <- get_dist(dt_rescaled, "euclidean")

hcl <- list()
for (linkage in c("average", "single", "complete", "ward")) {
  hcl[[linkage]] <- agnes(dist, method = linkage)
}

sapply(hcl, function(x) x$ac)
lapply(hcl, \(x) plot(x, which.plots = 2, main=paste0("Dendrogram for ", x$method, " linkage")))


### Cut the tree ----
cutcl_3 <- cutree(hcl[["ward"]], k = 3)
table(cutcl_3, labels)

### Analysis ----
fviz_cluster(list(data = dt,cluster = cutcl_3)) # if there are more than 2 dim, it uses PCA
# Here we see with all possible 2 dimensions
# The following can probably be done with facets. I 
combinations <- combn(colnames(dt), 2)
for(i in 1:ncol(combinations))
  print(fviz_cluster(list(data = dt, cluster = cutcl_3), choose.vars = combinations[,i]))

# VISUALIZE THE SILHOUETTE
fviz_silhouette(silhouette(cutcl_3, dist))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TWITTER DATASET ####
# ATTENTION: IT DOES NOT WORK ANYMORE. I KEEP IT FOR EDUCATIONAL PURPOSE ONLY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Tweets import ----
library(rtweet)

rtweet::auth_setup_default()
rt <- search_tweets("#siemens", n = 1000, include_rts = FALSE)
View(rt)
#saveRDS(rt, "../data/100_siemens_tweets.rds")
rt <- readRDS("./Lecture_08_100_siemens_tweets.rds")
View(rt)
## Build corpus ----
library(tm)
corpus <- Corpus(VectorSource(rt$text))
inspect(corpus)
content(corpus[1])

## Clean up corpus ----
newcorpus <- corpus
newcorpus <- tm_map(newcorpus, stripWhitespace)
newcorpus <- tm_map(newcorpus, removeWords, stopwords("en"))
newcorpus <- tm_map(newcorpus, removeWords, stopwords("de"))
newcorpus <- tm_map(newcorpus, tolower)
newcorpus <- tm_map(newcorpus, removeWords, "siemens")
newcorpus <- tm_map(newcorpus, removePunctuation)
# Or equivalently, with pipe notation
newcorpus <- corpus |>
  tm_map(stripWhitespace) |>
  tm_map(removeWords, stopwords("en")) |>
  tm_map(removeWords, stopwords("de")) |>
  tm_map(tolower) |>
  tm_map(removeWords, "siemens") |>
  tm_map(removePunctuation)

## Create TDM or DTM ----
#tdm <- TermDocumentMatrix(corpus, control = list(minWordLength=c(1,Inf)))
dtm <- DocumentTermMatrix(newcorpus, control = list(minWordLength=c(1,Inf)))
#tdm <- TermDocumentMatrix(corpus, control = list(minWordLength=c(1,Inf)))
inspect(dtm)
ncol(dtm)
View(as.matrix(dtm)[1:100, 1:100])


dtm <- removeSparseTerms(dtm, sparse = 0.99)
ncol(dtm)
# EDA
m<-as.matrix(dtm)
m[1:10, 1:10]

## Frequent terms ----
freq <- colSums(m)
str(freq)
freq[freq > 20]
plot_ly(x=names(freq[freq > 10]), y=freq[freq > 10],type = "bar") |>
  layout(xaxis=list(categoryorder = "array", categoryarray = names(sort(-freq[freq>10]))))

## Distance ----
dist_euc<-dist(m)
dist_euc_resc<-dist(scale(m))
dist_cos <- proxy::dist(m,method="cosine")

## Hierarchical ----
hcl <- list()
for (linkage in c("average", "single", "complete", "ward")) {
  hcl[[linkage]] <- agnes(dist_cos, method = linkage)
}

sapply(hcl, function(x) x$ac)
lapply(hcl, \(x) plot(x, which.plots = 2, main=paste0("Dendrogram for ", x$method, " linkage")))
abline(h=2.4)
fviz_dend(hcl[["ward"]], h=2.4)
hcl_cut <- cutree(hcl[["ward"]], h=2.4)

a <- data.table(tweet=content(newcorpus), cluster = hcl_cut)
View(a)

## Kmeans ----
cls <- data.table(k=seq(2,80,1), WSS=0)
for (i in cls[, k]) {
  cl <- kmeans(dist_cos, centers = i, nstart = 10)
  cls[k == i, WSS := cl$tot.withinss]
}
plot_ly(data=cls, x=~k, y=~WSS, type="scatter", mode="lines")
cl <- kmeans(dist_cos, centers = 10, nstart = 10)

a <- data.table(tweet=content(newcorpus), cluster = cl$cluster)
View(a[order(cluster)])


# doc a b c
# 1   0 1 1  -> 0.0 1.0 1.0
# 2   0 1 2  -> 0.0 0.5 1.0
# 3   0 0 0  -> 0.0 0.0 0.0
