from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X, y = datasets.load_iris(return_X_y=True)

X = X[:, 0].reshape(-1, 1)

m = LinearRegression()
m.fit(X, y)

y_pred = m.predict(X)

plt.scatter(X, y, color='blue', label="Actual")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression on Iris")
plt.legend()
plt.show()
