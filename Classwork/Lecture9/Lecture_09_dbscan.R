# Set up -------------------------------------------------------------------------------------------
# Working Directory
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Libraries
library(data.table)
library(plotly)
library(cluster) # additional clustering algorithms
library(factoextra) # A lot of nice visualizations and eval for clustering
#library(dendextend)
#library(fpc)
library(dbscan)

# SYNTHETIC dataset -------------------------------------------------------------------------------------
data("multishapes")
dt_src <- multishapes[, 1:3]
setDT(dt_src)
dt_src
plot_ly(data    = dt_src,
        x       = ~x,
        y       = ~y,
        color   = ~as.character(shape),
        symbol  = ~as.character(shape),
        type    = "scatter",
        mode    = "markers")

dt <- dt_src[, 1:2]
plot_ly(data    = dt,
        x       = ~x,
        y       = ~y,
        type    = "scatter",
        mode    = "markers")

## Kmeans ----
cls <- data.table(k = 1:10, WSS = 0)
for (i in cls[, k]) {
  cl <- kmeans(dt, centers = i, nstart = 25)
  cls[k == i, WSS := cl$tot.withinss]
}
plot_ly(data = cls, x = ~k, y = ~WSS, type = "scatter", mode = "lines")

cl_km <- kmeans(dt, centers = 2, nstart = 20)
plot_ly(data    = dt,
        x       = ~x,
        y       = ~y,
        color   = cl_km$cluster,
        symbol  = cl_km$cluster,
        type    = "scatter",
        mode    = "markers")

## Hclust ----
cl_hcl <- agnes(dt, method = "ward")
plot(cl_hcl, which.plots = 2)
cl_hcl <- cutree(cl_hcl, k = 5)
plot_ly(data    = dt,
        x       = ~x,
        y       = ~y,
        color   = cl_hcl,
        symbol  = cl_hcl,
        type    = "scatter",
        mode    = "markers")

## dbscan ----
# The two most popular packages for dbscan are "fpc" and "dbscan".
# fpc::dbscan is much slower than dbscan::dbscan. However, for our
# small dataset it does not make a difference.
# The library dbscan also contains more recent evolutions of dbscan (optics, hdbscan)

# Let's first plot the knn distance plot for the MinPTS rule of thumb (dim*2)
dbscan::kNNdistplot(dt, 4)
abline(h = 0.15, lty = 2)

cl_dbscan <- dbscan::dbscan(dt, eps = 0.15, minPts = 4)
cl_fpc <- fpc::dbscan(dt, eps = 0.15, MinPts = 4)

plot(cl_dbscan, data = dt)
plot(cl_fpc, data = dt)
plot_ly(data    = dt,
        x       = ~x,
        y       = ~y,
        color   = as.character(cl_dbscan$cluster),
        symbol  = as.character(cl_fpc$cluster),
        type    = "scatter",
        mode    = "markers")

## Hdbscan ----
cl_hdbscan <- hdbscan(dt, minPts = 4, gen_hdbscan_tree = T)
#cl.dbscan
plot(cl_hdbscan, scale = 10, show_flat = T)
View(cl_hdbscan)
plot_ly(x = dt$x,
        y = dt$y,
        color = as.character(cl_hdbscan$cluster),
        symbol = cl_hdbscan$cluster,
        type = "scatter",
        mode = "markers")

## OPTICS ----
cl_optics <- dbscan::optics(dt, minPts = 4)
plot(cl_optics)
plot(dt, col = "grey")
polygon(dt[cl_optics$order, ])
# Alternative 1: extract cluster by using a single threshold
cl_optics_static <- extractDBSCAN(cl_optics, eps_cl = 0.15)
plot(cl_optics_static)
# Alternative 2: extract cluster with varying density
# The current reachability plot is too unstable
# minPTS hase a smoothing effect
# We can increase minPts to smoothen a little the reachability
cl_optics <- dbscan::optics(dt, minPts = 10)
plot(cl_optics)

cl_optics_dyn <- extractXi(cl_optics, xi = 0.15)
plot(cl_optics_dyn)

# Clustering with eps-threshold
plot_ly(data    = dt,
        x       = ~x,
        y       = ~y,
        color   = cl_optics_static$cluster,
        symbol  = cl_optics_static$cluster,
        type    = "scatter",
        mode    = "markers")
# Clustering with Xi
plot_ly(data    = dt,
        x       = ~x,
        y       = ~y,
        color   = cl_optics_dyn$cluster,
        symbol  = cl_optics_dyn$cluster,
        type    = "scatter",
        mode    = "markers")

# Bank (Mixed data types) ----
#data <- fread("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
dt <- fread("./Lecture_09_bank.csv",
            select = c("age", "job", "marital", "education", "default", "balance", "housing"),
            stringsAsFactors = TRUE)
str(dt)
summary(dt)
dt[, default := ifelse(default == "yes", TRUE, FALSE)]
dt[, housing := ifelse(housing == "yes", TRUE, FALSE)]
dt[, education := ifelse(education == "unknown", NA, as.character(education))]
dt[, education := as.ordered(education)]

## Distance metric ----
# If you run it without weights, the distance will be dominated by the categorical features
gower_dist <- daisy(as.data.frame(dt),
                    metric = "gower",
                    type = list(asymm = c("default", "housing"), ordratio = "education"),
                    weights = c(1, 0.5, 0.4, 0.6, 1, 1, 0.5)
                    #weights = c(1,0.2,0.1,0.3,1,1,0.2)
)
summary(gower_dist)
# Let' create a matrix. It will come handy later
gower_mat <- as.matrix(gower_dist)
str(gower_mat)

# let's print the most different customers
idx_max <- which(gower_mat == max(gower_mat), arr.ind = T)
dt[c(idx_max[1, "row"], idx_max[1, "col"]), ]

# let's print the most similar customers
idx_min <- which(gower_mat == min(gower_mat[gower_mat != 0]), arr.ind = T)
dt[c(idx_min[1, "row"], idx_min[1, "col"]), ]

dt[, dist_from_first := gower_mat[, 1]]
View(dt[order(dist_from_first)])
dt[, dist_from_first := NULL]

#fviz_dist(gower_dist)
plot(density(gower_dist))
#plot(density(gower_mat[lower.tri(gower_mat)]))
dist_density <- density(gower_dist)
plot_ly(x       = dist_density$x,
        y       = dist_density$y,
        type    = "scatter",
        mode    = "line",
        fill    = "tozeroy")

##dbscan ----
kNNdistplot(gower_dist, k = 15)
# we see two knies. We can try both.
# Small eps will give us more clusters but compact, larger eps will give us less clusters less compact
abline(h = 0.05)
abline(h = 0.12)

cl_dbscan_05 <- dbscan(gower_dist, minPts = 15, eps = 0.05)
cl_dbscan_12 <- dbscan(gower_dist, minPts = 15, eps = 0.12)

lapply(unique(cl_dbscan_05$cluster), function(x) summary(dt[cl_dbscan_05$cluster == x]))
lapply(unique(cl_dbscan_12$cluster), function(x) summary(dt[cl_dbscan_12$cluster == x]))

## Hdbscan ----
cl_hdbscan <- hdbscan(gower_dist, minPts = 15, gen_simplified_tree = T)
plot(cl_hdbscan, scale = 1, show_flat = T)
View(cl_hdbscan)
lapply(unique(cl_hdbscan$cluster), function(x) summary(dt[cl_hdbscan$cluster == x]))

## OPTICS ----
cl_optics <- dbscan::optics(gower_dist, minPts = 15)
plot(cl_optics)
abline(h = 0.12)
abline(h = 0.08)

# Extract cluster by using a single threshold
cl_optics_static <- extractDBSCAN(cl_optics, eps_cl = 0.12)
plot(cl_optics_static)
lapply(unique(cl_optics_static$cluster), function(x) summary(dt[cl_optics_static$cluster == x]))


## Different weights (ToBeContinued) ----
gower_dist2 <- daisy(as.data.frame(dt), metric = "gower",
                    type = list(asymm = c("default", "housing"), ordratio = "education"),
                    #weights = c(1,0.5,0.4,0.6,1,1,0.5)
                    weights = c(1, 0.2, 0.1, 0.3, 1, 1, 0.2)
)

cl_optics2 <- dbscan::optics(gower_dist2, minPts = 15)
plot(cl_optics2)
abline(h = 0.07)
# Extract cluster by using a single threshold
cl_optics2_static <- extractDBSCAN(cl_optics2, eps_cl = 0.07)
plot(cl_optics2_static)
lapply(unique(cl_optics2_static$cluster), function(x) summary(dt[cl_optics_static$cluster == x]))
