from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# df = pd.read_csv("./Lecture_07_iris.csv")
# # drop the species column
# df = df.drop('Species', axis=1)
# print(df.head())
# # sccatter plot of the data
# # plt.scatter(df['Sepal.Length'], df['Sepal.Width'])
# # plt.xlabel('Sepal Length')
# # plt.ylabel('Sepal Width')
# # plt.show()

# # plt.scatter(df['Petal.Length'], df['Petal.Width'])
# # plt.xlabel('Petal Length')
# # plt.ylabel('Petal Width')
# # plt.show()

# k_list = np.arange(1, 10)
# SSE = []
# silhouette_scores = []
# for k in k_list:
#     kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300)
#     kmeans.fit(df)
#     SSE.append(kmeans.inertia_)
#     silhouette_scores.append(kmeans.score(df))

# # plot the SSE
# plt.plot(k_list, SSE, 'bx-')
# plt.xlabel('k')
# plt.ylabel('SSE')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

# # plot the silhouette scores
# plt.plot(k_list, silhouette_scores, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Silhouette Score')
# plt.title('The Silhouette Method showing the optimal k')
# plt.show()

# kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300).fit(df) # the optimal k is 3

### Image Segmentation

img = Image.open('./Lecture_07_willy.png')
img = img.resize((256, 256))
plt.imshow(img)
plt.show()

# convert the image to a numpy array
pixels = np.array(img)
pixels = pixels.reshape(-1, 3)
# SSE = []
# for k in k_list:
#     kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300)
#     kmeans.fit(pixels)
#     SSE.append(kmeans.inertia_)

# plt.plot(k_list, SSE, 'bx-')
# plt.xlabel('k')
# plt.ylabel('SSE')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

# the most optimal k is 3
kmeans = KMeans(n_clusters=10, n_init=10, max_iter=300).fit(pixels)
labels = kmeans.labels_
labels = labels.reshape((256, 256))
new_pixels = kmeans.cluster_centers_[labels]
new_img = Image.fromarray(new_pixels.astype('uint8'))
plt.imshow(new_img)
plt.show()
