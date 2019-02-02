import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

N = 500
k = 3

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2))

original_labels = np.array([0]*N + [1]*N + [2]*N)

def kmeans_display(X, labels):
  x0 = X[labels == 0, :]
  x1 = X[labels == 1, :]
  x2 = X[labels == 2, :]

  plt.plot(x0[:, 0], x0[:, 1], 'ro', markersize=2)
  plt.plot(x1[:, 0], x1[:, 1], 'g^', markersize=2)
  plt.plot(x2[:, 0], x2[:, 1], 'bs', markersize=2)
  plt.show()


def init_centers(X, k):
  return X[np.random.choice(X.shape[0], k, replace=False), :]

def assign_labels(X, centers):
  D = cdist(X, centers)
  return np.argmin(D, axis=1)

def update_centers(X, k, labels):
  newCenters = np.zeros((k, X.shape[1]))
  for k in range(k):
    newCenters[k] = np.mean(X[labels == k, :], axis=0)
  return newCenters

def has_coveraged(curCenters, newCenters):
  return (set([tuple(a) for a in curCenters]) == set([tuple(a) for a in newCenters]))

def kmeans(X, k):
  centers = [init_centers(X, k)]
  labels = []
  it = 0

  while True:
    labels.append(assign_labels(X, centers[-1]))
    newCenters = update_centers(X, k, labels[-1])
    if has_coveraged(centers[-1], newCenters):
      break
    centers.append(newCenters)
    it += 0
  return (centers, labels, it)

(centers, labels, it) = kmeans(X, k)

kmeans_display(X, labels[-1])

# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# predicted_labels = kmeans.predict(X)

# kmeans_display(X, predicted_labels)