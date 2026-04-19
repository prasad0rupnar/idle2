import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("voting_data.csv")
X = data[['Campaign_Spending', 'Approval_Rating']]
y = data['Result']
model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X)
plt.scatter(
data['Campaign_Spending'],
data['Approval_Rating'],
c=y,
cmap='bwr',
edgecolors='k'
)
plt.xlabel("Campaign Spending (Millions)")
plt.ylabel("Approval Rating (%)")
plt.title("Election Outcome Prediction (Logistic Regression)")
plt.show()
