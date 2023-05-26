import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Lecture_08_Mall_Customers.csv")
print(df.head( ))
print(df.describe( ))
print(df.info( ))
df = df.drop(["CustomerID"], axis=1)
# do a piplot with distribution of gender
# gender = df.groupby("Gender").size()
# plt.pie(gender, autopct='%1.1f%%', shadow=True, startangle=90)
# plt.title("Gender representation")
# plt.show( )

# plt.scatter(df["Spending Score (1-100)"], df["Annual Income (k$)"])
# plt.title("Spending Score vs Annual Income")
# plt.xlabel("Spending Score")
# plt.ylabel("Annual Income")
# plt.show()

# labels = df['Gender'].unique()
# data = [df[df['Gender'] == label]['Annual Income (k$)'] for label in labels]
# plt.boxplot(data, labels=labels)
# plt.title('Income by Gender')
# plt.ylabel('Income')
# plt.show()

# spending = data = [df[df['Gender'] == label]['Spending Score (1-100)'] for label in labels]
# plt.boxplot(spending, labels=labels)
# plt.title('Spending by Gender')
# plt.ylabel('Spending')
# plt.show()

# age = data = [df[df['Gender'] == label]['Age'] for label in labels]
# plt.boxplot(age, labels=labels)
# plt.title('Age by Gender')
# plt.ylabel('Age')
# plt.show()

hdf = df[["Annual Income (k$)", "Spending Score (1-100)"]]

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

for link in ["single", "complete", "average", "ward"]:
    Z = linkage(hdf, link)
    dendrogram(Z)
    plt.title("Dendrogram for {}".format(link))
    plt.show( )
    
# ward is the only one that makes sense
Z = linkage(hdf, "ward")
# cut the tree at the height of 200
labels = cut_tree(Z, height=200)
print(silhouette_score(hdf, labels))
plt.scatter(hdf["Annual Income (k$)"], hdf["Spending Score (1-100)"], c=labels)
plt.title("Clustering with Ward")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show( )
# silhouette score is 0.55, not bad
    