library(torch)

if (cuda_is_available()) {
    device <- torch_device("cuda")
} else {
    device <- torch_device("cpu")
}

# splitting IRIS
idx <- sample(nrow(iris), nrow(iris) * 0.8)

x_train <- as.matrix(iris[idx, 1:4])
y_train <- as.integer(iris[idx, 5])

x_test <- as.matrix(iris[-idx, 1:4])
y_test <- as.integer(iris[-idx, 5])

# convert to tensors
x_train_tensor <- torch_tensor(x_train, dtype = torch_float())
y_train_tensor <- torch_tensor(y_train, dtype = torch_long())

x_test_tensor <- torch_tensor(x_test, dtype = torch_float())
y_test_tensor <- torch_tensor(y_test, dtype = torch_long())

model <- nn_sequential( #4/8/16/3
    # layer 1
    nn_linear(4, 8), nn_relu(),
    # layer 2
    nn_linear(8, 16), nn_relu(),
    # layer 3
    nn_linear(16, 3), nn_softmax(dim = 2)
)

# Define cost function and optimizer
criterion <- nn_cross_entropy_loss()
optimizer <- optim_adam(model$parameters, lr = 0.005)

epochs <- 300

for (i in 1:epochs) {
    optimizer$zero_grad()

    # Forward pass
    y_pred_tensor <- model(x_train_tensor)

    # Compute loss
    loss <- criterion(y_pred_tensor, y_train_tensor)
    loss$backward()

    # take a step in the opposite direction
    optimizer$step()

    if (i %% 10 == 0) {
        winners <- y_pred_tensor$argmax(dim = 2)
        corrects <- winners == y_train_tensor
        accuracy <- corrects$sum()$item() / y_train_tensor$size()
        cat("Epoch:", i,
            "Loss", loss$item(),
            "Accuracy", accuracy, "\n")
    }
}

# Check on the test set
y_pred_tensor <- model(x_test_tensor)
y_pred <- as.array(y_pred_tensor$argmax(dim = 2))

print(table(y_pred, y_test))
cat(" Accuracy: ", sum(y_pred == y_test) / length(y_pred), "\n")

# # How to create a net by using modules (classes) instead of nn_sequential

# net <- nn_module(
#   "class_net",

#   initialize = function() {
#     self$linear1 <- nn_linear(4, 8)
#     self$linear2 <- nn_linear(8, 16)
#     self$linear3 <- nn_linear(16, 3)
#   },

#   forward = function(x) {
#     x |>
#     self$linear1() |>
#     nnf_relu() |>
#     self$linear2() |>
#     nnf_relu() |>
#     self$linear3() |>
#     nnf_softmax(dim = 2)
#   }
# )

# # Instantiate an object of the class
# model <- net()
