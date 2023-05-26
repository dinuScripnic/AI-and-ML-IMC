# SET UP -----------------------------------------------------------------------
# Working Directory
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Libraries
library(data.table)
library(plotly)
library(cluster) # additional clustering algorithms
library(factoextra) # A lot of nice visualizations and eval for clustering
library(corrplot)
library(ggplot2)
library(GGally)
library(imager) # For the second example


# IRIS DATASET -----------------------------------------------------------------
## DATA IMPORT ----
dt <- fread("./Lecture_07_iris.csv")
dt[, .N, Species]

# Let's assume we do not know the labels. But we save them for later checks
labels <- dt[, Species]
dt[, Species := NULL]

# Rescale (center around the mean and rescale by using stdev)
# dt.rescaled <- scale(dt)
# dt <- as.data.table(dt.rescaled)

## EDA ----
### PAIRPLOT ----
#### Base Graphics
graphics::pairs(dt, upper.panel = NULL)
#### Plotly
plotly::plot_ly(data = dt,
        type = "splom",
        dimensions = list(
          list(label='sepal length', values=~Sepal.Length),
          list(label='sepal width', values=~Sepal.Width),
          list(label='petal length', values=~Petal.Length),
          list(label='petal width', values=~Petal.Width))) |>
  #layout(plot_bgcolor='rgba(240,240,240, 0.95)') |>
  style(showupperhalf = F, diagonal = list(visible = F))
#### ggplot/GGally
GGally::ggpairs(dt)
#GGally::ggpairs(dt, aes(colour = labels))

### PCA ----
pca <- prcomp(dt, scale = T)
summary(pca)

# Screeplot: How much of the variance is explained by each component
fviz_screeplot(pca, addlabels = TRUE)
fviz_pca_var(pca, col.var = "contrib", repel = T)
# cos2: Square of the cosine of the angle between a variable and a PC
# How good a PC represents the variable
fviz_cos2(pca, choice = "var", axes = 1:2)
# contrib: cos2 of a variable / total cos2 of a component
# How much a variable contributes to a PC
fviz_contrib(pca, choice = "var", axes = 1:2)
fviz_pca_biplot(pca, col.var = "magenta", col.ind = labels, repel = T, addEllipses = T)
vars <- get_pca_var(pca)
plot_ly(x = colnames(vars$cos2), y = row.names(vars$cos2), z = vars$cos2, type = "heatmap", colors = "Reds")
plot_ly(x = colnames(vars$contrib), y = row.names(vars$contrib), z = vars$contrib, type = "heatmap", colors = "Reds")

plot_ly(data = dt, x = ~Sepal.Width, y = ~Sepal.Length, z = ~Petal.Width, color = labels,
        type = "scatter3d", mode = "markers")
plot_ly(x = pca$x[,"PC1"], y = pca$x[,"PC2"], z = pca$x[,"PC3"],
        type = "scatter3d", mode = "markers")

### DISTANCES ----
distance <- get_dist(dt, method = "euclidean")
fviz_dist(distance) # it points us to k=3


## CLUSTERING ----
# Remember, we do pretend not to know how many species we have.
# Let's assume we try with 5
cl <- kmeans(dt, centers = 5, nstart = 25)
cl
plot_ly(x = pca$x[,"PC1"], y = pca$x[,"PC2"], color = as.character(cl$cluster),
        type = "scatter", mode = "markers")

plot_ly(data = dt, x = ~Sepal.Width, y = ~Sepal.Length, z = ~Petal.Width, color = as.character(cl$cluster),
        type = "scatter3d", mode = "markers")
plot_ly(x = pca$x[,"PC1"], y = pca$x[,"PC2"], z = pca$x[,"PC3"], color = as.character(cl$cluster),
        type = "scatter3d", mode = "markers")


### Analysis ----

# VISUALIZE THE CLUSTERS
# VISUALIZE THE DISTANCE (in a sorted heatmap)
distance <- get_dist(dt, method = "euclidean")
fviz_cluster(cl, data = dt) # if there are more than 2 dim, it uses PCA
fviz_cluster(cl, data = distance) # if there are more than 2 dim, it uses PCA
# Here we see with all possible 2 dimensions
# The following can probably be done with facets.
combinations <- combn(colnames(dt), 2)
for (i in 1:ncol(combinations))
  print(fviz_cluster(cl, data = dt, choose.vars = combinations[,i]))

# VISUALIZE THE SILHOUETTE
fviz_silhouette(silhouette(cl$cluster, distance))

### Choose K ----
#with the Elbow method + silhouette score
cls <- data.table(k = 1:10, WSS = 0, SS = 0)
for (i in cls[, k]) {
  cl <- kmeans(dt, centers = i, nstart = 25)
  wss <- cl$tot.withinss
  ss <- ifelse(i != 1, mean(silhouette(cl$cluster, dist(dt))[, 3]), 0)
  cls[k == i, ':='(WSS = wss, SS = ss)]
}
plot_ly(data = cls, type = "scatter", mode = "lines") |>
  add_trace(x = ~k, y = ~WSS, name = "WSS") |>
  add_trace(x = ~k, y = ~SS, yaxis = "y2", name = "Silhouette Score", line = list(dash = "dash")) |>
  layout(yaxis2 = list(overlaying = "y", side = "right"))


### Final Clusters ----
# Elbow indicates a K=2 or K=3
cl <- kmeans(dt, centers = 3, nstart = 25)
fviz_cluster(cl, data = dt) # if there are more than 2 dim, it uses PCA
fviz_cluster(cl, data = distance) # if there are more than 2 dim, it uses PCA

cl <- kmeans(dt, centers = 2, nstart = 25)
fviz_cluster(cl, data = dt) # if there are more than 2 dim, it uses PCA
fviz_cluster(cl, data = distance) # if there are more than 2 dim, it uses PCA

# For learning purposes
library(animation)
kmeans.ani(dt, centers = 3)

# IMAGE DATASET ----------------------------------------------------------------

## Data import ----
img <- imager::load.image("./Lecture_07_willy.png")
plot(img)
str(img)
img_resized <- imager::resize(img, size_x = 128, size_y = 128)
plot(img_resized)

## Preprocessing ----
# Convert to data.table
dt_img <- as.data.frame(img_resized, wide = "c")
setDT(dt_img)
# If there is a transparency channel, we remove it
dt_img[, c.4:=NULL]
setnames(dt_img, c("c.1", "c.2", "c.3"), c("R", "G", "B"))

#convert back to cimg if necessary
#test<-as.cimg(unlist(newimg.dt[,3:5]), x=500, y=500, cc=3)
#plot(test)

# We can plot also with plotly
plot_ly(data = dt_img,
        x = ~x,
        y = ~y,
        type = "scattergl",
        mode = "markers",
        marker = list(color = ~rgb(R, G, B))) |>
  layout(yaxis = list(autorange = "reversed", scaleanchor = "x", scaleratio = 1))

## Clustering ----
dt_rgb <- dt_img[, .(R, G, B)]
## EDA ----
### How many unique colors?
dt_rgb[,.N,.(R,G,B)][order(N)]
uniqueN(dt_rgb[,.(R,G,B)])

### PAIRPLOT ----
GGally::ggpairs(dt_rgb)

### PCA ----
pca <- prcomp(dt_rgb, scale = T)
summary(pca)

# Screeplot: How much of the variance is explained by each component
fviz_screeplot(pca, addlabels = TRUE)
fviz_pca_var(pca, col.var = "cos2", repel = T)
# cos2: Square of the cosine of the angle between a variable and a PC
# How good a PC represents the variable
fviz_cos2(pca, choice = "var", axes = 1)
# contrib: cos2 of a variable / total cos2 of a component
# How much a variable contributes to a PC
fviz_contrib(pca, choice = "var", axes = 1)
#fviz_pca_biplot(pca, col.var = "magenta", repel = T, addEllipses = T)
vars <- get_pca_var(pca)
plot_ly(x = colnames(vars$cos2), y = row.names(vars$cos2), z = vars$cos2, type = "heatmap", colors = "Reds")
plot_ly(x = colnames(vars$contrib), y = row.names(vars$contrib), z = vars$contrib, type = "heatmap", colors = "Reds")

plot_ly(data = dt_rgb, x = ~R, y = ~G, z = ~B, type = "scatter3d", mode = "markers", marker = list(size = 2))
plot_ly(x = pca$x[,"PC1"], y = pca$x[,"PC2"], z = pca$x[,"PC3"],
        type = "scatter3d", mode = "markers", marker = list(size = 2))

### DISTANCES ----
#distance <- get_dist(dt_rgb, method = "euclidean")
#fviz_dist(distance) # This is slow. Do not run with large datasets


## Choose K ----
# Multiple k values
cls <- data.table(k = seq(2,40,2), WSS = 0)
for (i in cls[, k]) {
  cl <- kmeans(dt_rgb, centers = i, nstart = 30)
  wss <- cl$tot.withinss
  cls[k == i, WSS := wss]
}
plot_ly(data = cls, type = "scatter", mode = "lines") |>
  add_trace(x = ~k, y = ~WSS, name = "WSS")


## Final Clusters ----
# Elbow at ca. 8
km <- kmeans(dt_rgb, centers = 32, nstart = 20)
dt_newimg <- data.table(
  x = dt_img[, x],
  y = dt_img[, y],
  R = km$centers[km$cluster, "R"],
  G = km$centers[km$cluster, "G"],
  B = km$centers[km$cluster, "B"])

plot_ly(data = dt_newimg,
        x = ~x,
        y = ~y,
        type = "scattergl",
        mode = "markers",
        marker = list(color = ~rgb(R, G, B))) |>
  layout(yaxis = list(autorange = "reversed", scaleanchor = "x", scaleratio = 1))

fviz_cluster(km, data = dt_rgb) # if there are more than 2 dim, it uses PCA


