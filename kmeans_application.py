import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from display_network import *

mndata = MNIST('./MNIST/')
mndata.load_testing()
X = mndata.test_images

X0 = np.asarray(X)[:1000,:]
X = X0

K = 10
kmeans = KMeans(n_clusters=K).fit(X)

pred_label = kmeans.predict(X)

print(kmeans.cluster_centers_)
# print(pred_label)
# print(kmeans.cluster_centers_.shape)