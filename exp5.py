from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# load data
X, y = datasets.load_iris(return_X_y=True)

# take 2 classes and 2 features
X = X[y != 2][:, :2]
y = y[y != 2]

m = SVC(kernel="linear")
m.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = m.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='black', levels=[0])

plt.title("SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
