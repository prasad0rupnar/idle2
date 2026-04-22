import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# training data
X = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
y = np.array([1,-1,-1,1])

w = np.zeros(2)

for i in range(len(X)):
    w = w + X[i] * y[i]

print("Final Weights:", w)

sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette='coolwarm', s=100)

plt.title("Hebbian Learning")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
