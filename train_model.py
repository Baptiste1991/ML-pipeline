import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import pickle

# Load the dataset
df = pd.read_excel("insurance.xlsx")

#Checking data types 
print(df.dtypes)

# One-hot encode categorical variables (e.g., 'sex', 'smoker', 'region')
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Split the data into features (X) and target (y)
X = df.drop(columns='expenses')
y = df['expenses']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for numerical columns 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train the model 
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#prediction
y_pred = model.predict(X_test_scaled) 

# Model performance evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R2 Score: {r2:.2f}") 
print(f"mean absolute error: {mae:.2f}")
print(f"mean square error: {mse:.2f}")

# Save the model
with open('regressor_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training successful and saved as 'regressor_model.pkl'")
