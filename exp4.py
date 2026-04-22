from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# load data
d = datasets.load_iris()
X = d.data[:,:2]   # first 2 features
y = d.target

# model
m = LogisticRegression()
m.fit(X,y)

# plot
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
