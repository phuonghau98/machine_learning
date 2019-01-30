import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

one = np.ones((X.shape[0], 1))

xBar = np.concatenate((one, X), axis=1)

A = xBar.T.dot(xBar)
b = xBar.T.dot(y)

w = np.linalg.pinv(A).dot(b)

w0 = w[0][0]
w1 = w[1][0]

x0 = np.array([145, 185])
y0 = w1*x0 + w0

plt.plot(X.T, y.T, 'ro')
plt.plot(x0, y0)
plt.show()

print('Pretendable weight of people with height of 155 is: %.2fkg, real number: 52kg' %(155*w1 + w0))
print('Pretendable weight of people with height of 155 is: %.2fkg, real number: 56kg' %(160*w1 + w0))

# print('Scikit-learn lib')
# regr = linear_model.LinearRegression(fit_intercept=False)
# regr.fit(xBar, y)
# print('Solution found by scikit-learn:', regr.coef_)