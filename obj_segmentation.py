import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np

img = mpimg.imread('girl3.jpg')
# plt.imshow(img)
# plt.show()

X = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
cloneImg = np.zeros_like(X)

k = 3

kmeans = KMeans(n_clusters=k).fit(X)
pred_labels = kmeans.predict(X)
pred_centers = kmeans.cluster_centers_

for k in range(k):
    cloneImg[pred_labels == k] = pred_centers[k]

cloneImg = cloneImg.reshape((img.shape[0], img.shape[1], img.shape[2]))

print(pred_centers)

plt.imshow(cloneImg)
plt.show()
