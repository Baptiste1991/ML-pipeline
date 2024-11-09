import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def test_model_prediction(X_columns, scaler):
    # Example input data: a dictionary with sample values
    test_data = {
        "age": 30, 
        "sex": "male", 
        "bmi": 25, 
        "children": 2, 
        "smoker": "no", 
        "region": "southeast"
    }

    # Create DataFrame from test_data to match model input format
    test_df = pd.DataFrame([test_data])

    # One-hot encode categorical columns just like we did during training
    test_df = pd.get_dummies(test_df, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Ensure that the DataFrame has the same number of columns as the training data
    missing_cols = set(X_columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0

    # Make sure columns are in the same order as the training data
    test_df = test_df[X_columns]

    # Feature scaling using the same scaler used during training
    test_df_scaled = scaler.transform(test_df)

    # Load the saved model
    with open('regressor_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make prediction
    prediction = model.predict(test_df_scaled)

    # Assert the prediction is a numeric value (float)
    assert isinstance(prediction, np.ndarray), f"Prediction is not a valid ndarray, got {type(prediction)}"
    assert len(prediction) == 1, f"Prediction array length is incorrect: expected 1, got {len(prediction)}"
    assert isinstance(prediction[0], float), f"Prediction is not a float: got {type(prediction[0])}"

    print(f"Prediction: {prediction[0]:.2f}")

# Load the dataset and perform preprocessing (same as in your model training code)
df = pd.read_excel("insurance.xlsx")
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Split the data into features (X) and target (y)
X = df.drop(columns='expenses')
y = df['expenses']

# Feature scaling for numerical columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run the test
test_model_prediction(X.columns, scaler)
