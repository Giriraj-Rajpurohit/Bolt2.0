from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("forest.html", message=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    fever_fahrenheit = float(request.form['fever'])
    cough = request.form['cough']
    fatigue = request.form['fatigue']
    breathing = request.form['breathing']
    age = int(request.form['age'])
    gender = request.form['gender']

    # Convert Fever from Fahrenheit to binary (1 or 0)
    fever_binary = 1 if fever_fahrenheit >= 100 else 0

    # Convert categorical inputs to binary (1 or 0)
    cough_binary = 1 if cough == 'Yes' else 0
    fatigue_binary = 1 if fatigue == 'Yes' else 0
    breathing_binary = 1 if breathing == 'Yes' else 0
    gender_binary = 1 if gender == 'Male' else 0

    # Create input features array
    input_features = np.array([[fever_binary, cough_binary, fatigue_binary, breathing_binary, age, gender_binary]])

    # Make prediction using the model
    prediction = model.predict(input_features)

    # Generate message based on prediction
    if prediction == 1:
        message = 'We suggest you visit your doctor.'
    else:
        message = 'You do not need to visit the doctor. A simple call should suffice.'

    # Render the prediction message back to the user interface
    return render_template("forest.html", message=message)

if __name__ == '__main__':
    app.run(debug=True)
