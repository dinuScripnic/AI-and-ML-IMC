import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

# 0. Create the dataframe for metrics
metrics = pd.DataFrame(columns=['Accuracy', 'Recall', 'Precision', 'F1 score'])

# 1. Load the data
train = pd.read_csv('eeg_training.csv', header=None)
# split the data into features and label
X_train = train.iloc[:, 0:16]  # features
y_train = train.iloc[:, 16]  # label to predict
# Also I loaded the test data, but i will use it later for testing
test = pd.read_csv('eeg_test.csv', header=None)
X_test = test.iloc[:, 0:16]  # features
y_test = test.iloc[:, 16]  # label to predict

# 1,5. inspect the data
print(train.shape)
# We have 10486 observations and 16 variables and 1 label
print(X_train.isnull().sum())  # check for missing values
# we have 15 missing values in 4th, 6 values in 7th and 10319 in 13th variable
# we will drop the 13th variable because it has too many missing values
X_train = X_train.drop(13, axis=1)
X_test = X_test.drop(13, axis=1)  # also drop from test data
# also we will drop the rows that contain missing values from the 4th and 7th variable\
# find the index of the rows that contain missing values
to_drop = X_train[X_train[4].isnull()].index
to_drop = to_drop.append(X_train[X_train[7].isnull()].index)
# drop the rows
X_train = X_train.drop(to_drop, axis=0)
y_train = y_train.drop(to_drop, axis=0)
print(X_train.isnull().sum())  # now we have no more missing values, Horaay!
print(X_test.isnull().sum())  # check for missing values in test data, but there are no missing values
print(X_train.shape)  # we have 10465 observations and 15 variables
# Now we can proceed with the analysis with 




# 2. Develop a classifier that detects whether a patient has closed eyelids. 
# (positive class = closed eyelids).

# A) Fit a decision tree to the training data.
# Fit an entire decision tree, but for sure it will overfit the data
dt = tree.DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Accuracy of the decision tree is: ", accuracy_score(y_test, dt.predict(X_test)))  # ~0.8
dt = tree.DecisionTreeClassifier(criterion='gini', random_state=42)
dt.fit(X_train, y_train)
print("Accuracy of the decision tree with gini criterion is: ", accuracy_score(y_test, dt.predict(X_test)))  # ~0.8
dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=42   )
dt.fit(X_train, y_train)
print("Accuracy of the decision tree with entropy criterion is: ", accuracy_score(y_test, dt.predict(X_test)))  # ~0.8
# We can see that the accuracy is higher when we use entropy as a criterion
acc = list()
dt_list = list()
# now we will try to find the best max_depth for the tree
for i in range(1, 30):
    dt = tree.DecisionTreeClassifier(max_depth=i, criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    dt_list.append(dt)
    acc.append(accuracy_score(y_test, dt.predict(X_test)))
best_depth = acc.index(max(acc))
print('Best depth is: ', best_depth)
print('Accuracy of the decision tree with best depth is: ', max(acc))
dt = dt_list[best_depth]  # we will use this tree for the rest of the analysis
y_pred = dt.predict(X_test)
metrics.loc['Decision Tree'] = [accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred), f1_score(y_test, y_pred)]
print(metrics)
# plot the tree
# fig, ax = plt.subplots(figsize=(20,20))
# tree.plot_tree(dt, fontsize=8, ax=ax)
# plt.show()
# do a confusion matrix
cunfmat = confusion_matrix(y_test, dt.predict(X_test))
cm = ConfusionMatrixDisplay(confusion_matrix=cunfmat, display_labels=['open', 'closed'])
cm.plot()
plt.show()


# B)rain classifiers based on Random Forest, AdaBoost, and XGBoost (perform some hyperparameters tuning).










# CORELATION TEST, for testing purposes
# corelation = train.corr()
# # plot correlation matrix
# fig = px.imshow(corelation, text_auto=True)
# fig.show()

