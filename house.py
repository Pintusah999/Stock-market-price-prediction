from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Define the path to the model file inside the 'models' folder
model_path = os.path.join('models', 'house_price_prediction_model.pkl')

# Load the trained model from the models folder
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    city = request.form.get('city')
    location = request.form.get('location')
    area = float(request.form.get('area'))
    bedrooms = int(request.form.get('bedrooms'))
    gym = int(request.form.get('gym'))
    pool = int(request.form.get('pool'))
    
    # Make prediction
    prediction = predict_property_price(city, location, area, bedrooms, gym, pool)
    
    return prediction

def predict_property_price(city, location, area, bedrooms, gym, pool):
    # Example implementation
    # You can replace this with the actual logic for predicting property prices
    input_data = pd.DataFrame({
        'city': [city],
        'Location': [location],
        'Area': [area],
        'No. of Bedrooms': [bedrooms],
        'Gymnasium': [gym],
        'SwimmingPool': [pool],
        # Default values for other features
        '24X7Security': [0],
        'DiningTable': [0],
        'LandscapedGardens': [0],
        'Wifi': [0],
        'JoggingTrack': [0],
        'GolfCourse': [0],
        'MaintenanceStaff': [0],
        'StaffQuarter': [0],
        'VaastuCompliant': [0],
        'PowerBackup': [0],
        'Cafeteria': [0],
        'TV': [0],
        'ATM': [0],
        'LiftAvailable': [0],
        "Children'splayarea": [0],
        'School': [0],
        'Refrigerator': [0],
        'IndoorGames': [0],
        'MultipurposeRoom': [0],
        'RainWaterHarvesting': [0],
        'Intercom': [0],
        'Wardrobe': [0],
        'WashingMachine': [0],
        'Hospital': [0],
        'ShoppingMall': [0],
        'CarParking': [0],
        'Gasconnection': [0],
        'SportsFacility': [0],
        'ClubHouse': [0],
        'Microwave': [0],
        'BED': [0],
        'Sofa': [0],
        'Resale': [0],
        'AC': [0],
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
