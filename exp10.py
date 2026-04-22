from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# load data
d = datasets.load_iris()
X = d.data
y = d.target

# apply PCA
p = PCA(n_components=2)
X_pca = p.fit_transform(X)

# explained variance
print("Explained Variance Ratio:", p.explained_variance_ratio_)

# plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Iris Dataset")

# legend
plt.legend(*scatter.legend_elements(), title="Classes")

plt.grid(True)
plt.show()
