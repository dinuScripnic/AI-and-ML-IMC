# TODO: import willing to present
# NOTE: I want to present NOTE:
import pandas as pd
import numpy as np
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def print_parameters(parameters: dict) -> str:
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
train = pd.read_csv(
    "https://onedrive.live.com/download?resid=4C66E14E953F6D39!9922&authkey=!AJL6R66cYHl1E0Q",
    header=None,
)
# Also I loaded the test data, but i will use it later for testing
test = pd.read_csv(
    "https://onedrive.live.com/download?resid=4C66E14E953F6D39!9921&authkey=!AP0enoDbayF1mbE",
    header=None,
)

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
# ---------------------Decision Tree---------------------#
# Now lets search for the best tree parameters
dt_parameters = {
    "max_depth": [10, 11, 12, 13, 14, 15],
    "min_samples_split": [5, 10, 15],
    "criterion": ["entropy", "gini"],
}  # set the parameters for the grid search

# use grid search to find the best parameters
grid_dt = GridSearchCV(
    estimator=dt, param_grid=dt_parameters, cv=5, scoring="accuracy", n_jobs=-1
)  # create the grid search
# cv stays for the number of cross validation folds, n_jobs is the number of jobs to run in parallel (most of computers have at least 4 cores, so i pick 2 to be safe)
# important to mention that the grid will maximize the accuracy on training data, it will not be best for test
# also it takes a while to run, invest into i9-13900K, it costs more than my entire laptop...
grid_dt.fit(X_train, y_train)  # fit the grid search
best_dt: tree.DecisionTreeClassifier = grid_dt.best_estimator_  # get the best estimator
print(
    "Best parameters for Decision Tree are: ", print_parameters(best_dt.get_params())
)  # print the best parameters
dt_pred = best_dt.predict(X_test)  # predict the labels for the test data
dt_pred_train = best_dt.predict(X_train)  # predict the labels for the train data
metrics.loc["Decision Tree"] = [
    accuracy_score(y_test, dt_pred),  # aprox 0.797
    recall_score(y_test, dt_pred),  # aprox 0.844
    precision_score(y_test, dt_pred),  # aprox 0.799
    f1_score(y_test, dt_pred),  # aprox 0.821
]  # add the metrics to the dataframe
metrics.loc["Decision Tree train"] = [
    accuracy_score(y_train, dt_pred_train),  # aprox 0.997
    recall_score(y_train, dt_pred_train),  # aprox 0.997
    precision_score(y_train, dt_pred_train),  # aprox 0.997
    f1_score(y_train, dt_pred_train),  # aprox 0.997
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
# B)train classifiers based on Random Forest, AdaBoost, and XGBoost (perform some hyperparameters tuning).

# ------------------------Random Forest------------------------
# ----My Random Forest----#
# rf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_features='log2', oob_score=True, n_jobs=2, random_state=42)
# to get this parameters i was using the brain and try and error method, i tried to find the best parameters for the model
# other parameters rather then this ones were killing the performance of the model, but i believe its because i am an idiot
##########################
# Now lets search for the best forest parameters
rf_params: dict[str:list] = {
    "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "criterion": ["entropy", "gini"],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [10, 11, 12, 13, 14, 15],
    "min_samples_split": [5, 10, 15],
}  # set the parameters for the search
rf = RandomForestClassifier(
    n_jobs=-1,
    oob_score=True,  # out of bag score
    random_state=42,  # random state
)  # create the random forest, i will use the random search to find the best parameters
# why random state 42? because its the answer to the ultimate question of life, the universe, and everything
# use random search to find the best parameters, its faster than grid search, but it doesn't guarantee to find the best parameters
# faster, relative to grid search, because it doesn't try all the combinations of parameters, but it still takes AGES,
# i did notes for risk management and i am still waiting for the results
rsc = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    n_iter=10,  # number of iterations, but this is quite low, i would use much more but it takes too long, my laptop sees Musk's tesla
)  # create the random search
rsc.fit(X_train, y_train)  # fit the model
best_rf: RandomForestClassifier = rsc.best_estimator_  # get the best estimator
rf_pred = best_rf.predict(X_test)  # predict the labels for the test data
rf_pred_train = best_rf.predict(X_train)  # predict the labels for the train data
print(
    "Best parameters for Random Forest are: ", print_parameters(best_rf.get_params())
)  # print the best parameters
metrics.loc["Random Forest"] = [
    accuracy_score(y_test, rf_pred),  # aprox 0.899
    recall_score(y_test, rf_pred),  # aprox 0.941
    precision_score(y_test, rf_pred),  # aprox 0.883
    f1_score(y_test, rf_pred),  # aprox 0.911
]  # add the metrics to the dataframe, the best i could get
metrics.loc["Random Forest train"] = [
    accuracy_score(y_train, rf_pred_train),  # aprox 0.997
    recall_score(y_train, rf_pred_train),  # aprox 0.997
    precision_score(y_train, rf_pred_train),  # aprox 0.997
    f1_score(y_train, rf_pred_train),  # aprox 0.997
]  # add the metrics to the dataframe

cm2 = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, rf_pred),
    display_labels=["open", "closed"],
)

# plot the feature importances
importances = pd.Series(
    best_rf.feature_importances_, index=X_train.columns
)  # get the feature importances
importances = importances.sort_values(ascending=False)  # sort the feature importances
# plot the feature importances
imp_plt = importances.plot(kind="bar", color="gray")  # plot the feature importances
imp_plt.set_xlabel("Features")  # set the x label
imp_plt.set_ylabel("Importance")  # set the y label
imp_plt.set_title("Feature Importances for Random Forest")  # set the title
plt.show()
# we can mention that most of the data has some importance, only 11th column has a lower importance than others
# considering the importance graph, in the future we maybe can drop 11, because it has a big drop in importance compared to the others

# ----AdaBoost----#
ada = AdaBoostClassifier(
    estimator=best_dt, n_estimators=500, random_state=42
)  # create the AdaBoost

ada.fit(X_train, y_train)  # fit the AdaBoost
print("Best parameters for AdaBoost are: ", print_parameters(ada.get_params()))
ada_pred = ada.predict(X_test)  # predict the labels for the test data
# create the confusion matrix, will be used later
ada_pred_train = ada.predict(X_train)  # predict the labels for the train data
cm3 = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, ada_pred),
    display_labels=["open", "closed"],
)
importances = pd.Series(
    ada.feature_importances_, index=X_train.columns
)  # get the feature importances
importances = importances.sort_values(ascending=False)  # sort the feature importances
imp_plt = importances.plot(kind="bar", color="gray")  # plot the feature importances
imp_plt.set_xlabel("Features")  # set the x label
imp_plt.set_ylabel("Importance")  # set the y label
imp_plt.set_title("Feature Importances for Ada Boost")  # set the title
plt.show()

metrics.loc["AdaBoost"] = [
    accuracy_score(y_test, ada_pred),  # aprox 0.904
    recall_score(y_test, ada_pred),  # aprox 0.952
    precision_score(y_test, ada_pred),  # aprox 0.888
    f1_score(y_test, ada_pred),  # aprox 0.918
]  # add the metrics to the dataframe
metrics.loc["AdaBoost train"] = [
    accuracy_score(y_train, ada_pred_train),  # aprox 0.997
    recall_score(y_train, ada_pred_train),  # aprox 0.997
    precision_score(y_train, ada_pred_train),  # aprox 0.997
    f1_score(y_train, ada_pred_train),  # aprox 0.997
]  # add the metrics to the dataframe
 # print the time it took to run the AdaBoost
############################################################################################################
# ----XGBoost----#
y_train_1 = y_train.replace({1: 0, 2: 1})  # replace the labels with 1 and 0
y_test_1 = y_test.replace({1: 0, 2: 1})  # replace the labels with 1 and 0
# this is because xgboost is able to handle only binary labels
xgboost = xgb.XGBClassifier(
    n_estimators=1000, max_depth=16, min_child_weight=5, random_state=42, n_jobs=2
)
xgboost.fit(X_train, y_train_1)  # fit the model
xgboost_pred = xgboost.predict(X_test)  # predict the labels for the test data
xgboost_pred_train = xgboost.predict(X_train)  # predict the labels for the train data
metrics.loc["XGBoost"] = [
    accuracy_score(y_test_1, xgboost_pred),  # aprox 0.919
    recall_score(y_test_1, xgboost_pred),  # aprox 0.908
    precision_score(y_test_1, xgboost_pred),  # aprox 0.911
    f1_score(y_test_1, xgboost_pred),  # aprox 0.91
]  # add the metrics to the dataframe
metrics.loc["XGBoost train"] = [
    accuracy_score(y_train_1, xgboost_pred_train),  # aprox 0.997
    recall_score(y_train_1, xgboost_pred_train),  # aprox 0.997
    precision_score(y_train_1, xgboost_pred_train),  # aprox 0.997
    f1_score(y_train_1, xgboost_pred_train),  # aprox 0.997
]  # add the metrics to the dataframe
# create the confusion matrix, it will be used later
cm4 = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test_1, xgboost_pred),
    display_labels=["open", "closed"],
)

############################################################################################################

# ----Confusion Matrix----#
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs[0, 0].imshow(cm1.confusion_matrix, cmap=cmap)
axs[0, 0].set_title("Decision Tree")
axs[0, 1].imshow(cm2.confusion_matrix, cmap=cmap)
axs[0, 1].set_title("Random Forest")
axs[1, 0].imshow(cm3.confusion_matrix, cmap=cmap)
axs[1, 0].set_title("AdaBoost")
axs[1, 1].imshow(cm4.confusion_matrix, cmap=cmap)
axs[1, 1].set_title("XGBoost")
# TODO: create better labels and legend
color_bar = fig.colorbar(cm1.im_, ax=axs)
plt.legend()
plt.show()  # show the plot
############################################################################################################


# ----Results----#
print(metrics)  # print the metrics

############################################################################################################
