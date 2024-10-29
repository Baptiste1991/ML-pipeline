import unittest
import pickle
import pandas as pd
from app import app

# Load the trained logistic regression model
model = pickle.load(open('loan_model.pkl', 'rb'))

class TestLoanPredictionModel(unittest.TestCase):

    # Test the model's prediction accuracy
    def test_model_prediction(self):
        # Sample input for testing as DataFrame (with feature names)
        input_data = pd.DataFrame([[30, 70000]], columns=['age', 'salary'])
        prediction = model.predict(input_data)

        # Check if the prediction is either 0 or 1
        self.assertIn(prediction[0], [0, 1], "Prediction should be 0 or 1.")

    # Test the Flask app with JSON response
    def test_flask_app(self):
        # Set up a test client
        tester = app.test_client(self)

        # Define sample input data as JSON
        input_data = {'age': 30, 'salary': 70000}

        # Make a POST request to the /predict endpoint with JSON headers
        response = tester.post('/predict', json=input_data)

        # Check the response status code
        self.assertEqual(response.status_code, 200)

        # Check if the response contains 'payback' key
        response_data = response.get_json()
        self.assertIsNotNone(response_data, "Response should not be None.")
        self.assertIn('payback', response_data, "Response should contain 'payback' key.")

if __name__ == '__main__':
    unittest.main()