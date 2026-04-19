import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Load dataset
iris = datasets.load_iris()

# Use only two classes and two features
X = iris.data[iris.target != 2][:, :2]
y = iris.target[iris.target != 2]

# Train SVM model
model = SVC(kernel="linear")
model.fit(X, y)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y)

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)

xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy)
Z = Z.reshape(XX.shape)

ax.contour(XX, YY, Z, levels=[0], colors="black")

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("SVM Decision Boundary for Iris Classification")
plt.show()
