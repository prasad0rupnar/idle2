import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# create dataset inside code
data = pd.DataFrame({
    "Years_of_Education": [10, 12, 14, 16, 18, 20, 22],
    "Income": [20000, 25000, 30000, 40000, 50000, 65000, 80000]
})

# features and target
X = data[['Years_of_Education']]
y = data['Income']

# model
model = LinearRegression()
model.fit(X, y)

# prediction
y_pred = model.predict(X)

# plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel('Years of Education')
plt.ylabel('Income')
plt.title('Education vs Income (Linear Regression)')
plt.legend()
plt.show()
