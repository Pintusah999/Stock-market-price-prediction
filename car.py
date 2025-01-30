from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# Define the path to the model file inside the 'models' folder
model_path = os.path.join('models', 'car_price_model_simple.pkl')

# Load the trained model from the models folder
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        features = request.form.to_dict()

        # Validate and parse the input values, providing default values if necessary
        input_data = {
            'symboling': [float(features.get('symboling', 0))],
            'fueltype': [features.get('fueltype', 'gas')],
            'aspiration': [features.get('aspiration', 'std')],
            'carbody': [features.get('carbody', 'sedan')],
            'drivewheel': [features.get('drivewheel', 'fwd')],
            'enginesize': [float(features.get('enginesize', 0))],
            'horsepower': [float(features.get('horsepower', 0))],
            'citympg': [float(features.get('citympg', 0))],
            'highwaympg': [float(features.get('highwaympg', 0))],
            'curbweight': [float(features.get('curbweight', 0))]
        }

        # Convert input data to a pandas DataFrame
        input_df = pd.DataFrame(input_data)

        # Debugging logs (optional)
        print("Input DataFrame:")
        print(input_df)

        # Predict car price
        prediction = model.predict(input_df)[0]

        # Return the prediction in a JSON response
        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        # Handle any errors gracefully and return a JSON response with the error message
        return jsonify({'error': f"Error: {str(e)}"})

def predict_vehicle_price(features):
    try:
        # Validate and parse the input values, providing default values if necessary
        input_data = {
            'symboling': [float(features.get('symboling', 0))],
            'fueltype': [features.get('fueltype', 'gas')],
            'aspiration': [features.get('aspiration', 'std')],
            'carbody': [features.get('carbody', 'sedan')],
            'drivewheel': [features.get('drivewheel', 'fwd')],
            'enginesize': [float(features.get('enginesize', 0))],
            'horsepower': [float(features.get('horsepower', 0))],
            'citympg': [float(features.get('citympg', 0))],
            'highwaympg': [float(features.get('highwaympg', 0))],
            'curbweight': [float(features.get('curbweight', 0))]
        }

        # Convert input data to a pandas DataFrame
        input_df = pd.DataFrame(input_data)

        # Predict car price
        prediction = model.predict(input_df)[0]

        return round(prediction, 2)

    except Exception as e:
        raise ValueError(f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
