from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('regressor_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        
        # One-hot encode the region based on user input
        region = request.form['region']
        region_northwest = 1 if region == 'northwest' else 0
        region_southeast = 1 if region == 'southeast' else 0
        region_southwest = 1 if region == 'southwest' else 0

        # Construct the feature array in the order expected by the model
        features = np.array([[age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest]])

        # Make the prediction
        prediction = model.predict(features)

        # Render the form again with the prediction result
        return render_template('index.html', prediction=f"{prediction[0]:.2f}")

    except Exception as e:
        return render_template('index.html', prediction="An error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)