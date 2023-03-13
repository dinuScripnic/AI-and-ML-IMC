if (rstudioapi::isAvailable())
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(data.table)
library(plotly)
library(caret)

data <- fread('./wines.csv')
head(data)
dim(data)
str(data)
summary(data)
# data represents 4 columns and 1599 rows
# we will try to predict the quality of the wine based on the other 3 columns
# plot the wine quality


# divide the data into training and testing sets
set.seed(123)
sample <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.8, 0.2))
train  <- data[sample, ]
test   <- data[!sample, ]

# fit a multiple linear regression model
model <- lm(quality ~ density, volatileacidity, data = train)
summary(model)
train_pred <- predict(model, train)
train_actual <- train$quality
MAE_train <- mean(abs(train_pred - train_actual))
MAEP_train <- mean(abs(train_pred - train_actual)/train_actual)
RMSE_train <- sqrt(mean((train_pred - train_actual)^2))
R2_train <- 1 - sum((train_pred - train_actual)^2)/sum((train_actual - mean(train_actual))^2)
test_pred <- predict(model, test)
test_actual <- test$quality
MAE_test <- mean(abs(test_pred - test_actual))
MAEP_test <- mean(abs(test_pred - test_actual)/test_actual)
RMSE_test <- sqrt(mean((test_pred - test_actual)^2))
R2_test <- 1 - sum((test_pred - test_actual)^2)/sum((test_actual - mean(test_actual))^2)

# structure MAE, MAEP, RMSE, R2 for training and testing sets
MLR_results <- data.frame(c(MAE_train, MAEP_train, RMSE_train, R2_train), c(MAE_test, MAEP_test, RMSE_test, R2_test))

# fit a polynomial regression model
MAEs_training <- vector()
MAEs_test <- vector()
for (i in 1:8) {
    poly_regr<- lm(quality ~ poly(density, chlorides, volatileacidity, degree = i), data = train)

    train_pred <- poly_regr$fitted.values
    train_actual <- train$quality
    MAEs_training[i] <- MAE(train_pred, train_actual)

    test_pred <- predict(poly_regr, test)
    test_actual <- test$quality
    MAEs_test[i] <- MAE(test_pred, test_actual)
}
plot_ly(x = 1:length(MAEs_training),y = MAEs_training, type = "scatter", mode = "line") %>%
  add_lines(x = 1:length(MAEs_test), y = MAEs_test)


# the model starts to overfit after degree 4
# fit a polynomial regression model with degree 4

poly_regr<- lm(quality ~ poly(density, chlorides, volatileacidity, degree = 4), data = train)
train_pred <- poly_regr$fitted.values
train_actual <- train$quality
MAE_train <- mean(abs(train_pred - train_actual))
MAEP_train <- mean(abs(train_pred - train_actual)/train_actual)
RMSE_train <- sqrt(mean((train_pred - train_actual)^2))
R2_train <- 1 - sum((train_pred - train_actual)^2)/sum((train_actual - mean(train_actual))^2)
test_pred <- predict(poly_regr, test)
test_actual <- test$quality
MAE_test <- mean(abs(test_pred - test_actual))
MAEP_test <- mean(abs(test_pred - test_actual)/test_actual)
RMSE_test <- sqrt(mean((test_pred - test_actual)^2))
R2_test <- 1 - sum((test_pred - test_actual)^2)/sum((test_actual - mean(test_actual))^2)
PR_results <- data.frame(first_row = c('MAE', 'MAEP', 'RMSE', 'R2'),train =  c(MAE_train, MAEP_train, RMSE_train, R2_train), test = c(MAE_test, MAEP_test, RMSE_test, R2_test), )

PR_results
