import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Read dataset
iris = pd.read_csv('./Lecture_10_iris.csv')

# Label encoding
iris['Species'] = iris['Species'].astype('category').cat.codes

x_train, x_test, y_train, y_test = train_test_split(iris[iris.columns[0:4]].values,
                                                    iris.Species.values, test_size=0.2)

# convert to tensors
x_train_tensor = torch.Tensor(x_train).float()
y_train_tensor = torch.Tensor(y_train).long()

x_test_tensor = torch.Tensor(x_test).float()
y_test_tensor = torch.Tensor(y_test).long()


model = nn.Sequential(
    # layer 1
    nn.Linear(4, 8), nn.ReLU(),  # 4/8/16/3
    # layer 2
    nn.Linear(8, 16), nn.ReLU(),  # 4/8/16/3
    # layer 3
    nn.Linear(16, 3), nn.Softmax(dim = 1)  # 4/8/16/3
)

# Define cost function and optimizer
criterion = nn.CrossEntropyLoss()  # combines nn.LogSoftmax() and nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # lr = learning rate

epochs = 300  # number of times to loop through the entire dataset

for i in range(epochs):  # loop through the entire dataset a number of times
    optimizer.zero_grad()  # zero the gradients

    # Forward pass: Compute predicted y by passing  x to the model
    y_pred_tensor = model(x_train_tensor)  # model.forward(x_train_tensor)

    # Compute loss
    loss = criterion(y_pred_tensor, y_train_tensor)
    loss.backward()
    
    # take a step in the opposite direction
    optimizer.step()

    if i % 10 == 0:
        winners = y_pred_tensor.argmax(dim=1)
        print('Epoch:', i, 
              'Loss:', loss.item(), 
              'Accuracy:', accuracy_score(winners, y_train_tensor))


y_pred_tensor=model(x_test_tensor)
# print(y_pred_tensor)

y_pred = y_pred_tensor.argmax(1)
# _, y_pred = y_pred_tensor.max(1)

print('\nConfusion Matrix (Test Set):\n', confusion_matrix(y_pred, y_test))
print('\nAccuracy (Test Set):', accuracy_score(y_test_tensor, y_pred))

# class Net(nn.Module):
#   def __init__(self, n_input, n_layer1, n_layer2, n_output):
#     super().__init__()
#     self.hid1 = nn.Linear(n_input, n_layer1)  # 4/8/16/3
#     self.relu1 = nn.ReLU()
#     self.hid2 = nn.Linear(n_layer1, n_layer2)
#     self.relu2 = nn.ReLU()
#     self.oupt = nn.Linear(n_layer2, n_output)
#     self.softmax = nn.Softmax(dim=1)

#   def forward(self, x):
#     z = self.hid1(x)
#     z = self.relu1(z)
#     z = self.hid2(z)
#     z = self.relu2(z)
#     z = self.oupt(z)  
#     z = self.softmax(z)
#     return z

# model = Net(n_input = 4, n_layer1 = 8, n_layer2 = 16, n_output = 3)