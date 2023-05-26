import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt

df = pd.read_csv("Lecture_04_Banknotes.txt", sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

print(df.isna().sum())  #  there are no missing values
print(df.shape)  #  1372 rows and 5 columns
print(df.dtypes)  # variables are float and class is int


# fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)

# for i, col in enumerate(df.columns[:-1]):
#     axs[i % 2].scatter(x=range(len(df)), y=df[col], c=df['class'], cmap='viridis')
#     axs[i % 2].set_title(col)

# plt.tight_layout()
# plt.show()

# 1. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[["variance", "skewness", "curtosis", "entropy"]], df["class"], test_size=0.2, random_state=42
)


dt = tree.DecisionTreeClassifier(random_state=42, criterion='entropy')  # wtf the accuracy is already 97 percent
dt.fit(X_train, y_train)
print(dt.get_params())
print(dt.score(X_test, y_test))


# fit a grid search model to the data
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_leaf_nodes': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}
# how many combinations are there?
# 2 * 9 * 9 * 10 * 11 = 1, 98, 400
# gsc = GridSearchCV(
#     estimator=tree.DecisionTreeClassifier(),
#     param_grid=params,
#     cv=5, scoring='accuracy', verbose=0, n_jobs=-1, random_state=42
# )
# gsc.fit(X_train, y_train)   
# best_params = gsc.best_params_
# print(best_params)  
# best_dt = gsc.best_estimator_
# print(best_dt.score(X_test, y_test))
# what does n_jobs -1 do?
# n_jobs=-1 means using all processors



rsc = RandomizedSearchCV(
    estimator=tree.DecisionTreeClassifier(),
    param_distributions=params,
    cv=5, scoring='accuracy', verbose=0, n_jobs=-1, n_iter=1000, random_state=42
)   
rsc.fit(X_train, y_train)
print(rsc.best_params_)
print(rsc.best_estimator_.score(X_test, y_test))


