# Banknotes data
# 1. variance of Wavelet Transformed image (continuous)
# 2. skewness of Wavelet Transformed image (continuous)
# 3. curtosis of Wavelet Transformed image (continuous)
# 4. entropy of image (continuous)
# 5. class (0 is genuine, 1 is forged)

# Set WD to the folder of the currently opened r file
if (rstudioapi::isAvailable())
  setwd(dirname(rstudioapi::getActiveDocumentContext()[["path"]]))

# Load libraries
library(data.table) # Data wrangling (Py_equivalent: pandas, polars, pydatatable)
library(plotly) # Beautiful plots
library(rpart) # Decision trees
library(rpart.plot)
library(caret) # ML tools http://topepo.github.io/caret/available-models.html
#library(mlr3) # ML tools (Py_equivalent: Scikit learn)

# Load the data ----
dt <- fread("./Lecture_04_Banknotes.txt")
setnames(dt,
         old = c("V1","V2","V3","V4","V5"),
         new = c("Variance", "Skewness", "Curtosis", "Entropy", "Class")) # rename columns
dt[, Class := as.factor(Class)] # convert to categorical

# or combined all in one fread
dt <- fread("./Lecture_04_Banknotes.txt",
            col.names = c("Variance", "Skewness", "Curtosis", "Entropy", "Class"),
            colClasses = c("numeric","numeric","numeric","numeric","factor"))  # set column types

summary(dt)  # look at the data
dt[, Class := as.factor(ifelse(Class == "0", "Genuine", "Forged"))]

# Explore the data ----
colSums(is.na(dt))
# rowSums(is.na(dt))
# any(is.na(dt))

# Look at each variable and the class (Scatterplot)
a <- plot_ly(data = dt, x = 1:nrow(dt), y = ~Variance, color = ~Class,
        type = "scatter", mode = "markers", showlegend = F)
b <- plot_ly(data = dt, x = 1:nrow(dt), y = ~Skewness, color = ~Class,
        type = "scatter", mode = "markers", showlegend = F)
c <- plot_ly(data = dt, x = 1:nrow(dt), y = ~Curtosis, color = ~Class,
        type = "scatter", mode = "markers", showlegend = F)
d <- plot_ly(data = dt, x = 1:nrow(dt), y = ~Entropy, color = ~Class,
        type = "scatter", mode = "markers")
subplot(a,b,c,d, nrows = 2, titleY = T, margin = 0.05)

# better do this in a loop
plots <- list()
for (i in colnames(dt[, !"Class"])) {
  plots[[i]] <- plot_ly(data = dt, x = 1:nrow(dt), y = dt[,get(i)], color = dt[,Class],
                        type = "scatter", mode = "markers")
}
subplot(plots, nrows = 2, titleY = T, margin = 0.05)

# let's fix the legend and add the axis title
plots <- list()
for (i in colnames(dt[, !"Class"])) {
  show_legend <- ifelse(i == "Variance", TRUE, FALSE)
  plots[[i]] <- plot_ly(data = dt, x = 1:nrow(dt), y = dt[,get(i)], color = dt[,Class],
                        type = "scatter", mode = "markers", showlegend = show_legend)  |>
    layout(yaxis = list(title = i))
}
subplot(plots, nrows = 2, titleY = T, margin = 0.05)

# For those who already know a bit of R
# This could have been done also with lapply
#
# plots <- list()
# plots <- lapply(colnames(dt[, !"Class"]), function(i) {
#     show_legend <- ifelse(i == "Variance", TRUE, FALSE)
#     plot_ly(dt, x = 1:nrow(dt), y = ~get(i), color = ~Class,
#             type = "scatter", mode = "markers", showlegend = show_legend) |>
#     layout(yaxis = list(title = i))
#   }
# )
# subplot(plots, nrows = 2, titleY = T, margin = 0.05)

# Look at the boxplots
plots <- list()
for(i in colnames(dt[,!"Class"])){
  show_legend <- ifelse(i == "Variance", TRUE, FALSE)
  plots[[i]] <- plot_ly(data = dt, x = dt[,Class], y = dt[,get(i)], color = dt[,Class],
                        type = "box", showlegend = show_legend) |>
    layout(yaxis = list(title = i))
}
subplot(plots, nrows = 2, titleY = T, margin = 0.05)

# It's ordered, we should shuffle it first.
set.seed(12345)
dt <- dt[sample(1:nrow(dt))]

# Look if class is distinguishable from with the three most important variables
plot_ly(data = dt,
        x = ~Variance, y = ~Skewness, color = ~Class,
        type = "scatter", mode = "markers", marker = list(size = 4))
plot_ly(data = dt,
        x = ~Variance, y = ~Skewness, z = ~Curtosis, color = ~Class,
        type = "scatter3d", mode = "markers", marker = list(size = 4))

# Splom (aka scatter matrix, aka pair plot)
plot_ly(data = dt, type = 'splom', color = ~Class, marker = list(size = 4),
    dimensions = list(
      list(label='Variance', values=~Variance),
      list(label='Skewness', values=~Skewness),
      list(label='Curtosis', values=~Curtosis),
      list(label='Entropy', values=~Entropy)))

# Split training/test ----
set.seed(12345)
idx <- createDataPartition(dt[,Class], p = 0.8, list = F, times = 1)
training <- dt[idx]
test <- dt[!idx]
# dt[where, select, by]
training[, .(n_obs = .N, share = .N / nrow(training)), Class] 

test[, .(n_obs = .N, share = .N / nrow(test)), Class]

test_x <- test[,!"Class"]
test_y <- test[,Class]


# Train a classification Tree ----
## OPTION 1: Stopping criterion ----
# Grow the tree until stopping criterion is met
# Default rpart parameters: minsplit = 20 and cp = 0.01
# IMPORTANT: rpart already includes 10-fold-validation internally
# and show the cross-validation error with the printcp() function

fit <- rpart::rpart(Class ~ ., method = "class", data = training) 
print(fit)
printcp(fit)
plotcp(fit)
summary(fit)
rpart.plot(fit, type = 2, extra = 101, fallen.leaves = F, main = "Classification Tree for Banknotes", tweak=1.2)

## OPTION 2: Entire + pruning ----
# Grow the entire tree, until we see it overfitting and then "prune" it
# (the previous tree did not overfit until cp=0.01)
fit.entire <- rpart::rpart(Class ~ ., method = "class", data = training,
                           control = rpart.control(minsplit = 1, cp = 0))  # proceed untill i reach the minimum number of observations in each node, until 100% accuracy on training set
print(fit.entire)
printcp(fit.entire)
plotcp(fit.entire)
summary(fit.entire)
rpart.plot(fit.entire, type = 2, extra = 101, fallen.leaves = F, tweak = 1.2, main = "Entire tree for Banknotes")

# And now we prune it at the optimal level of CP
best_cp_for_pruning <- fit.entire$cptable[which.min(fit.entire$cptable[, "xerror"]), "CP"]  # goes till the very end and we cut it at the optimal point
fit.entire.pruned <- prune(fit.entire, cp = best_cp_for_pruning)

# This is our final tree
fit.entire.pruned
printcp(fit.entire.pruned)
rpart.plot(fit.entire.pruned, type = 2, extra = 101, fallen.leaves = F, tweak = 1.2, main = "Pruned tree for Banknotes")

# Predict ----
# Initial tree with default values
my_pred_initial_tree <- predict(fit, newdata = test_x, type = "class")
my_pred_entire_tree <- predict(fit.entire, newdata = test_x, type = "class")
my_pred_pruned_tree <- predict(fit.entire.pruned, newdata = test_x, type = "class")


table(my_pred_initial_tree,test_y)
confusionMatrix(my_pred_initial_tree, reference = test_y)
confusionMatrix(my_pred_initial_tree, reference = test_y, positive = "Forged")
confusionMatrix(my_pred_initial_tree, reference = test_y, positive = "Forged", mode = "prec_recall")
confusionMatrix(my_pred_entire_tree,test_y, positive = "Forged", mode = "prec_recall")
confusionMatrix(my_pred_pruned_tree, test_y, positive = "Forged", mode = "prec_recall")

# Classification Tree with Caret----
# Important Caret aspects:
# trainControl
# tuneGrid
# tuneLength
# train(..., )
# Start
ctrl <- trainControl(method = "cv", number = 10)
# ctrl <- trainControl(method = "repeatedcv",
#                      number = 10,
#                      repeats = 3)
# ctrl <- trainControl(method="LOOCV")
fit.caret <- train(Class ~ ., data = training, method = "rpart", trControl = ctrl, tuneLength = 10)
# available methods and parameters are at http://topepo.github.io/caret/available-models.html
my_pred <- predict(fit.caret, newdata = test_x)
table(my_pred,test_y)
confusionMatrix(data = my_pred, reference = test_y, positive = "Forged", mode = "prec_recall")
summary(fit.caret)
rpart.plot(fit.caret$finalModel)
