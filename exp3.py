import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv("education_income.csv")
X = data[['Years_of_Education']]
y = data['Income']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
e
plt.xlabel('Years of Education')
plt.ylabel('Income')
plt.title('Education vs Income (Linear Regression)')
plt.legend()
plt.show()
