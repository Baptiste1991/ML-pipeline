from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained logistic regression model
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Render HTML template

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Determine if the request is sending JSON data
        if request.is_json:
            data = request.get_json()
            age = int(data['age'])
            salary = int(data['salary'])
        else:
            # Otherwise, get data from the form input
            age = int(request.form['age'])
            salary = int(request.form['salary'])

        # Prepare input for the model as a DataFrame with feature names
        input_data = pd.DataFrame([[age, salary]], columns=['age', 'salary'])

        # Make prediction
        prediction = model.predict(input_data)

        # Determine the prediction result
        result = 'Will Pay Back Loan' if prediction[0] == 1 else 'Will Not Pay Back Loan'

        # Check if the request expects a JSON response
        if request.is_json:
            return jsonify({'payback': int(prediction[0])}), 200

        # Otherwise, render the result on the HTML page
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        # Return JSON error for API calls
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        
        # Render error message on the HTML page for form submissions
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
