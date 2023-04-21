from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('./Lecture_06_BreastCancerCoimbra.csv')
# print how many none values are in each column
# print(df.isnull().sum()) # no missing values
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
k_list = np.arange(1, 85, 2)
accuracies = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    accuracies.append(score)
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

    