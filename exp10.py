import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=[
    "sepal length", "sepal width", "petal length", "petal width", "target" ])
features = ["sepal length", "sepal width", "petal length", "petal width"]
x = df.loc[:, features].values
y = df.loc[:, ["target"]].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(
    data=principalComponents,
    columns=["principal component 1", "principal component 2"] )
finalDf = pd.concat([principalDf, df[["target"]]], axis=1)
plt.figure(figsize=(8, 8))
target_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2 }
colors_num = df["target"].map(target_mapping)
plt.scatter(
    principalComponents[:, 0],
    principalComponents[:, 1],
    c=colors_num,
    cmap="plasma")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.show()
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
