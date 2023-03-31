import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import pycaret


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


# ------------------------------------Load Data---------------------------------------------#
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
X_train, y_train = clean_data(train)
X_test, y_test = clean_data(test)
############################################################################################


# ---------------------------------------kNN------------------------------------------------#
index = np.arange(1, 20, 2)
test_accuracy = np.array([])
train_accuracy = np.array([])
# loop through the odd numbers from 1 to 20 and find the best number of neighbors
for i in np.arange(1, 20, 2):
    # Create the model
    model = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    # Train the model
    model.fit(X_train, y_train)
    # Save the accuracy
    train_accuracy = np.append(train_accuracy, model.score(X_train, y_train))
    test_accuracy = np.append(test_accuracy, model.score(X_test, y_test))
df = pd.DataFrame(
    {
        "Number of neighbors": index,
        "Accuracy on train": train_accuracy,
        "Accuracy on test": test_accuracy,
    }
)  # create a dataframe with the results
# plot the accuracy to find the most optimal number of neighbors
fig = px.line(
    df, x="Number of neighbors", y=["Accuracy on train", "Accuracy on test"]
)  # create the plot
fig.update_traces(mode="markers+lines")  # add the markers and the lines
fig.update_layout(
    title="Accuracy of the model",
    xaxis_title="Number of neighbors",
    yaxis_title="Accuracy",
)  # add the title and the axis labels
fig.show()  # show the plot
# the best accuracy is achieved at 7 neighbors and it is 91.1%

# now we will find the best metric
metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
out = []
for metric in metrics:
    model = KNeighborsClassifier(n_neighbors=7, metric=metric, n_jobs=-1)
    model.fit(X_train, y_train)
    out.append((metric, model.score(X_test, y_test)))
# sort the list by the accuracy
out.sort(key=lambda x: x[1], reverse=True)
print(out)
best_knn = KNeighborsClassifier(n_neighbors=7, metric=out[0][0], n_jobs=-1)
best_knn.fit(X_train, y_train)
pred = best_knn.predict(X_test)
print(
f"""
Accuracy: {accuracy_score(pred, y_test)}
Precision: {precision_score(pred, y_test)}
Recall: {recall_score(pred, y_test)}
F1: {f1_score(pred, y_test)}
"""
)
print("The best model is the KNN with 7 neighbors and the metric", out[0][0])

########################################################################################


# ------------------------------------CARET---------------------------------------------#
