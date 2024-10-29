import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
    'payback': [0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)


# Features and target
X = df[['age', 'salary']]
y = df['payback']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Save the model
with open('loan_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training successful and saved as 'loan_model.pkl'")
