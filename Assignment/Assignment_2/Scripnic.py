import pandas as pd
import numpy as np
import itertools
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def print_parameters(parameters: dict[str : int | float | str | None]) -> str:
    """
    Return a string with the parameters in a nice format.
    Args:
        parameters (dict): dictionary with the parameters
    Returns:
        str: string with the parameters
    """
    out: str = str("\n")  # create the string
    for key, value in parameters.items():  # iterate through the parameters
        out += f"\t{key}: {value}\n"  # add the parameter to the string
    return out  # return the string


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.core.series.Series]:
    """
    Clean the data by removing the missing values.
    Args:
        data (pd.DataFrame): dataframe with the data

    Returns:
        tuple[pd.DataFrame, pd.core.series.Series]: tuple with the cleaned data, first element is the dataframe with the variables and the second element is the series with the labels
    """
    data = data.drop(
        13, axis=1
    )  # we are removing column 13 because it has 10k missing values
    data = data.dropna(axis=0)  # we are removing all rows with missing values
    X = data.iloc[:, 0:15]  # we are creating a dataframe with all the variables
    y = data.iloc[:, 15]  # we are creating a series with the labels
    return X, y  # return the cleaned data


colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)

# 0. Create the dataframe for metrics
metrics = pd.DataFrame(columns=["Accuracy", "Recall", "Precision", "F1 score"])

# 1. Load the data
train = pd.read_csv("eeg_training.csv", header=None)
# Also I loaded the test data, but i will use it later for testing
test = pd.read_csv("eeg_test.csv", header=None)

# 1,5. inspect the data
# print(train.shape)
# print(test.shape)
# # We have 10486 observations and 16 variables and 1 label
# print(
#     train.isnull().sum()
# )  # we have 15 missing values in column 4, 6 in column 7 and 10k in column 13
# print(test.isnull().sum())  # we don't have any missing values in test data
X_train, y_train = clean_data(train)  # clean the data
X_test, y_test = clean_data(test)  # clean the data
# print(X_train.isnull().sum())  # now we have no more missing values, Horaay!
# print(X_train.shape)  # we have 10465 observations and 15 variables
############################################################################################################

# 2. Develop a classifier that detects whether a patient has closed eyelids.
# (positive class = closed eyelids).

# A) Fit a decision tree to the training data.
dt = tree.DecisionTreeClassifier()  # create the decision tree

# Now lets search for the best tree parameters
dt_parameters = {
    "max_depth": [10, 11, 12, 13, 14, 15],
    "min_samples_split": [5, 10, 15],
    "criterion": ["entropy", "gini"],
}  # set the parameters for the grid search

# use grid search to find the best parameters
grid_dt = GridSearchCV(
    estimator=dt, param_grid=dt_parameters, cv=5, scoring="accuracy", n_jobs=2
)  # create the grid search
# cv stays for the number of cross validation folds, njobs is the number of jobs to run in parallel (most of computers have at least 4 cores, so i pick 2 to be safe)
# important to mention that the grid will maximize the accuracy on training data, it will not be best for test
# also it takes a while to run, invest into i9-13900K, it costs more than my entire laptop...
grid_dt.fit(X_train, y_train)  # fit the grid search
best_dt: tree.DecisionTreeClassifier = grid_dt.best_estimator_  # get the best estimator
print(
    "Best parameters for Decision Tree are: ", print_parameters(best_dt.get_params())
)  # print the best parameters
dt_pred = best_dt.predict(X_test)  # predict the labels for the test data
metrics.loc["Decision Tree"] = [
    accuracy_score(y_test, dt_pred),  # aprox 0.8
    recall_score(y_test, dt_pred),  # aprox 0.85
    precision_score(y_test, dt_pred),  # aprox 0.8
    f1_score(y_test, dt_pred),  # aprox 0.82
]  # add the metrics to the dataframe

# Plot the decision tree
# fig, ax = plt.subplots(figsize=(20, 20))
# tree.plot_tree(dt, fontsize=8, ax=ax)
# plt.show()
# I commented this out for now, because it takes a lot of time to run, and its too big to understand something from it

# Do a confusion matrix, because why not
cm1 = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, dt_pred),
    display_labels=["open", "closed"],
)

plt.show()


# B)rain classifiers based on Random Forest, AdaBoost, and XGBoost (perform some hyperparameters tuning).

# Random Forest
# Now lets search for the best tree parameters
# n_estimators = [100,200,500,1000]
# max_depth = [10, 11, 12, 15]
# min_samples_split = [5, 10, 15]
# conbos = list(itertools.product(n_estimators, max_depth, min_samples_split))
# results = []
# for conbo in conbos:
#     rf = RandomForestClassifier(n_estimators=conbo[0], max_depth=conbo[1], min_samples_split=conbo[2])
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     results.append([conbo, accuracy_score(y_test, y_pred)])
# this takes ages, so ill use the parameters i found
rf = RandomForestClassifier(
    n_estimators=500,
    criterion="entropy",
    max_features="log2",
    random_state=42,
    oob_score=True,
    n_jobs=2,
)
# to get this parameters i was using the brain and try and error method, i tried to find the best parameters for the model
# other parameters rather then this ones were killing the performance of the model, but i believe its because i am an idiot
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Best parameters for Random Forest are: ", print_parameters(rf.get_params()))
metrics.loc["Random Forest"] = [
    accuracy_score(y_test, rf_pred),  # aprox 0.899
    recall_score(y_test, rf_pred),  # aprox 0.941
    precision_score(y_test, rf_pred),  # aprox 0.883
    f1_score(y_test, rf_pred),  # aprox 0.911
]  # add the metrics to the dataframe, the best i could get

cm2 = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, rf_pred),
    display_labels=["open", "closed"],
)

# plot the feature importances
importances = pd.Series(
    rf.feature_importances_, index=X_train.columns
)  # get the feature importances
importances = importances.sort_values(ascending=False)  # sort the feature importances
importances.plot(kind="bar")  # plot the feature importances
plt.show()
# we can mention that most of the data has some importance, only 11th column has a lower importance than others
# considering the importance graph, in the future we maybe can drop 11, because it has a big drop in importance compared to others


# AdaBoost
dt = tree.DecisionTreeClassifier(
    max_depth=16, min_samples_split=5, random_state=42, criterion="entropy"
)  # create the decision tree
ada = AdaBoostClassifier(estimator=dt, n_estimators=500)  # create the AdaBoost
ada.fit(X_train, y_train)  # fit the AdaBoost
print("Best parameters for AdaBoost are: ", print_parameters(ada.get_params()))
ada_pred = ada.predict(X_test)  # predict the labels for the test data
cm3 = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, ada_pred),
    display_labels=["open", "closed"],
)

metrics.loc["AdaBoost"] = [
    accuracy_score(y_test, ada_pred),  # aprox 0.8
    recall_score(y_test, ada_pred),  # aprox 0.85
    precision_score(y_test, ada_pred),  # aprox 0.8
    f1_score(y_test, ada_pred),  # aprox 0.82
]  # add the metrics to the dataframe


# XGBoost
y_train_1 = y_train.replace({1: 0, 2: 1})  # replace the labels with 1 and 0
y_test_1 = y_test.replace({1: 0, 2: 1})  # replace the labels with 1 and 0
# this is because xgboost is able to handle only binary labels
xgboost = xgb.XGBClassifier(
    n_estimators=1000, max_depth=16, min_child_weight=5, random_state=42, n_jobs=2
)
xgboost.fit(X_train, y_train_1)  # fit the model
xgboost_pred = xgboost.predict(X_test)  # predict the labels for the test data
metrics.loc["XGBoost"] = [
    accuracy_score(y_test_1, xgboost_pred),  # aprox 0.9
    recall_score(y_test_1, xgboost_pred),  # aprox 0.94
    precision_score(y_test_1, xgboost_pred),  # aprox 0.89
    f1_score(y_test_1, xgboost_pred),  # aprox 0.91
]  # add the metrics to the dataframe
cm4 = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test_1, xgboost_pred),
    display_labels=["open", "closed"],
)

# plot all the confusion matrices together
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs[0, 0].imshow(cm1.confusion_matrix, cmap=cmap)
axs[0, 0].set_title("Decision Tree")
axs[0, 1].imshow(cm2.confusion_matrix, cmap=cmap)
axs[0, 1].set_title("Random Forest")
axs[1, 0].imshow(cm3.confusion_matrix, cmap=cmap)
axs[1, 0].set_title("AdaBoost")
axs[1, 1].imshow(cm4.confusion_matrix, cmap=cmap)
axs[1, 1].set_title("XGBoost")
plt.legend()
print(metrics)  # print the metrics
plt.show()  # show the plot
############################################################################################################
