import base64
import json
import os
import pickle

import cohere
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from werkzeug.security import check_password_hash, generate_password_hash

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

app.secret_key = '\xa3k\x1e)\x1c\xa8-m\x8e4\x01;\xcc!\xefn\xb7\xdd\xe05\xabq\xec>'

# Initialize Cohere Client
COHERE_API_KEY = '55FarykCm1MeYkxvqnbOIuz4Ec4bljh1mlZ5kqFP'  # Replace with your Cohere API key
co = cohere.Client(COHERE_API_KEY)

# Paths to models
stock_model_path = os.path.join('models', 'lstm_stock_price_model.h5')
stock_scaler_path = os.path.join('models', 'scaler.pkl')
car_model_path = os.path.join('models', 'car_price_model_simple.pkl')
house_model_path = os.path.join('models', 'house_price_prediction_model.pkl')

# Load car and house models
try:
    car_model = joblib.load(car_model_path)
    house_model = joblib.load(house_model_path)
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")

# --- Stock Prediction Functions ---
def load_stock_model_and_scaler():
    try:
        stock_model = load_model(stock_model_path)
        with open(stock_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return stock_model, scaler
    except Exception as e:
        print(f"Error loading stock model or scaler: {e}")
        return None, None

def fetch_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            raise ValueError("No data found for the given ticker.")
        return data[['Close']]
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def prepare_data_for_prediction(data, time_step=60):
    model, scaler = load_stock_model_and_scaler()
    if model is None or scaler is None:
        return None, None
    try:
        scaled_data = scaler.transform(data[['Close']].values)
        X = []
        X.append(scaled_data[-time_step:].reshape(-1, 1))
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, scaler
    except Exception as e:
        print(f"Error preparing data for prediction: {e}")
        return None, None

def predict_future_prices(model, data, days=30):
    X, scaler = prepare_data_for_prediction(data)
    if X is None or scaler is None:
        return []
    future_prices = []
    try:
        for _ in range(days):
            predicted_price = model.predict(X)
            future_prices.append(predicted_price[0][0])
            X = np.append(X[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
        future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
        return future_prices.flatten()
    except Exception as e:
        print(f"Error predicting future prices: {e}")
        return []

def generate_graphs(data, predicted_prices, ticker):
    try:
        fig_actual = go.Figure()
        fig_actual.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price'))
        fig_actual.update_layout(
            title=f'Actual Stock Price for {ticker}',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark'
        )
        img_actual = fig_actual.to_html(full_html=False)

        fig_predicted = go.Figure()
        fig_predicted.add_trace(go.Scatter(x=list(range(1, 31)), y=predicted_prices, mode='lines', name='Predicted Price'))
        fig_predicted.update_layout(
            title=f'Predicted Stock Price for {ticker}',
            xaxis_title='Days (1-30)',
            yaxis_title='Price',
            xaxis=dict(tickmode='array', tickvals=list(range(1, 31))),
            template='plotly_dark'
        )
        img_predicted = fig_predicted.to_html(full_html=False)

        return img_actual, img_predicted
    except Exception as e:
        print(f"Error generating graphs: {e}")
        return "", ""

# --- Routes ---
@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')



@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        # Handle form submission (e.g., save user to the database)
        username = request.form['username']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256', salt_length=8)

        if User.query.filter_by(username=username).first():
            return "Username already exists!"

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        return redirect('/login')

    # Render registration page for GET request
    return render_template('register.html')



@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/stock', methods=['GET', 'POST'])
def stock():
    if request.method == 'POST':
        ticker = request.form.get('ticker')  # Get ticker from POST form data
        if not ticker:
            return render_template('stock.html', error="Please enter a stock ticker.")
        data = fetch_stock_data(ticker)
        if data.empty:
            return render_template('stock.html', error="Invalid ticker or no data found.")
        model, scaler = load_stock_model_and_scaler()
        if model is None or scaler is None:
            return render_template('stock.html', error="Error loading prediction model.")
        predicted_prices = predict_future_prices(model, data)
        
        # Check if predicted_prices is valid (e.g., not empty or None)
        if predicted_prices is None or len(predicted_prices) == 0:
            return render_template('stock.html', error="Error making predictions.")
        
        # Convert predicted_prices to a list for JSON serialization
        session['predicted_prices'] = predicted_prices.tolist()  # Convert ndarray to list
        session['ticker'] = ticker
        
        img_base64_actual, img_base64_predicted = generate_graphs(data, predicted_prices, ticker)
        predicted_data = [(f"Day {i+1}", price) for i, price in enumerate(predicted_prices)]
        
        return render_template('stock.html',
                               ticker=ticker,
                               img_base64_actual=img_base64_actual,
                               img_base64_predicted=img_base64_predicted,
                               predicted_data=predicted_data)
    return render_template('stock.html')

@app.route('/ai-suggestion', methods=['POST'])
def ai_suggestion():
    data = request.json
    ticker = data.get('ticker', '').upper()

    if not ticker:
        return jsonify({'error': 'Ticker is required.'}), 400

    # Check if a prediction has been made and is stored in the session
    if 'predicted_prices' not in session or session.get('ticker') != ticker:
        return jsonify({'error': 'Please predict the price first before generating an AI suggestion.'}), 400

    # Fetch historical data for context
    historical_data = fetch_stock_data(ticker)
    if historical_data is None or historical_data.empty:
        return jsonify({'error': 'Unable to fetch historical data.'}), 500

    # Retrieve the predicted prices from the session
    predicted_prices = session['predicted_prices']
    
    # Convert the list back to a NumPy array if necessary
    predicted_prices = np.array(predicted_prices)

    # Prepare the input for AI
    past_prices = historical_data['Close'].tail(30).tolist()  # Last 30 days of historical prices
    prediction_str = ', '.join([f"Day {i+1}: ${price:.2f}" for i, price in enumerate(predicted_prices)])

    # Build a detailed prompt for the AI model
    prompt = f"""
    Provide a detailed analysis and actionable advice for the stock ticker {ticker}.
    The past 30 days of stock prices were: {', '.join([f"${price:.2f}" for price in past_prices])}.
    The predicted prices for the next 30 days are: {prediction_str}.
    Analyze these data points to determine if the stock is currently a good buy, sell, or hold.
    Include insights based on historical trends, price movements, and the predicted future values.
    Give your response in plain text with approximately 100 words and it should end with a fullstop.
    """

    # Query Cohere API
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=250,
        temperature=0.7,  # Adjust for creativity
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )

    # Extract and return the generated suggestion
    generated_text = response.generations[0].text.strip()
    return jsonify({'output': generated_text})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        # Get the user input from the request
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        # Generate a response using Cohere or any other logic
        prompt = f"User: {user_input}\nBot:"
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=prompt,
            max_tokens=100,  # Small output
            temperature=0.7,  # Adjust creativity
            k=0,
            stop_sequences=["\n"]
        )

        # Extract the generated text
        bot_output = response.generations[0].text.strip()

        # Return the response
        return jsonify({'bot_reply': bot_output})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/car', methods=['GET', 'POST'])
def car():
    if request.method == 'POST':
        features = request.form.to_dict()
        action = request.form.get('action', 'predict')  # Default to prediction action
        
        try:
            # Prepare input data
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
            input_df = pd.DataFrame(input_data)

            if action == 'ai_suggestion':
                # Check if prediction is made
                if 'prediction' not in session:
                    return render_template('car.html', error="Please predict the price first before using AI Suggestion.")
                
                # Use the existing prediction in the AI suggestion prompt
                predicted_price = session['prediction']
                suggestion_input = (
                    f"Given the car details below:\n"
                    f"Attributes: {input_data}\n"
                    f"Predicted Price: ${predicted_price}\n\n"
                    f"Provide an analysis of whether this car is a good deal at the predicted price. "
                    f"Include reasons why someone might or might not consider purchasing this car. "
                    f"Give the response in plain text with no bullet points or list tags and limit to 200 words.and give ans in 150 word only okay and it should end with fullstop"
                )
                
                # Generate AI suggestion using Cohere API
                ai_suggestion = co.generate(
                    model="command-xlarge-nightly",
                    prompt=suggestion_input,
                    max_tokens=200,
                    temperature=0.7
                ).generations[0].text

                # Pass the AI suggestion and prediction to the template
                return render_template('car.html', ai_suggestion=ai_suggestion.strip(), prediction=predicted_price)

            # Prediction logic
            prediction_car = car_model.predict(input_df)[0]
            session['prediction'] = round(prediction_car, 2)  # Save prediction in session
            return render_template('car.html', prediction=round(prediction_car, 2))
        except Exception as e:
            print("Error during car processing:", str(e))
            return render_template('car.html', error=f"Error: {str(e)}")
    
    return render_template('car.html')

@app.route('/house', methods=['GET', 'POST'])
def house():
    if request.method == 'POST':
        try:
            # Debugging print to check form data
            print("Form data received:", request.form)

            # Collect form data
            city = request.form.get('city')
            location = request.form.get('location')
            area = float(request.form.get('area'))
            bedrooms = int(request.form.get('bedrooms'))
            gym = int(request.form.get('gym'))
            pool = int(request.form.get('pool'))

            # Create input DataFrame
            input_data = {
                'city': [city],
                'Location': [location],
                'Area': [area],
                'No. of Bedrooms': [bedrooms],
                'Gymnasium': [gym],
                'SwimmingPool': [pool]
            }

            # Add missing columns with default value 0
            missing_columns = [
                '24X7Security', 'DiningTable', 'LandscapedGardens', 'LiftAvailable', 'AC', 
                'Microwave', 'IndoorGames', 'Sofa', 'StaffQuarter', "Children'splayarea", 
                'Intercom', 'WashingMachine', 'CarParking', 'JoggingTrack', 'PowerBackup', 
                'Wardrobe', 'Gasconnection', 'VaastuCompliant', 'GolfCourse', 'MaintenanceStaff', 
                'Cafeteria', 'SportsFacility', 'Resale', 'ShoppingMall', 'Hospital', 'ClubHouse', 
                'School', 'ATM', 'MultipurposeRoom', 'Wifi', 'BED', 'Refrigerator', 'TV', 
                'RainWaterHarvesting'
            ]
            for column in missing_columns:
                input_data[column] = [0]

            input_df = pd.DataFrame(input_data)

            # Make prediction
            prediction = house_model.predict(input_df)[0]
            print("Prediction result:", prediction)

            # Send data (without predicted price) to the AI suggestion endpoint
            response = requests.post(
                'http://localhost:5000/house-ai-suggestion',
                json={
                    'city': city,
                    'location': location,
                    'area': area,
                    'bedrooms': bedrooms,
                    'gym': gym,
                    'pool': pool
                }
            )
            
            # Check if the response is successful
            if response.status_code == 200:
                ai_suggestion = response.json().get('output', 'No suggestion available.')
                return render_template('house.html', prediction=round(prediction, 2), ai_suggestion=ai_suggestion)
            else:
                return render_template('house.html', error="Failed to get AI suggestion.")
        
        except Exception as e:
            print("Error during house prediction:", str(e))
            return render_template('house.html', error=f"Error: {str(e)}")
    return render_template('house.html')

@app.route('/house-ai-suggestion', methods=['POST'])
def house_ai_suggestion():
    if request.method == 'POST':
        try:
            # Collect JSON data
            data = request.get_json()
            city = data.get('city')
            location = data.get('location')
            area = data.get('area')
            bedrooms = data.get('bedrooms')
            gym = data.get('gym')
            pool = data.get('pool')

            # Check for missing or invalid inputs
            if not city or not location or not area or not bedrooms or not gym or not pool:
                return jsonify({"error": "All fields are required."})

            # Convert types as necessary
            area = float(area)
            bedrooms = int(bedrooms)
            gym = int(gym)
            pool = int(pool)

            # Construct the prompt for Cohere's API
            prompt = f"""
            Provide a detailed analysis of the real estate market for a house with the following attributes:
            - City: {city}
            - Location: {location}
            - Area: {area} sq. ft.
            - Number of Bedrooms: {bedrooms}
            - Availability of Gym: {'Yes' if gym else 'No'}
            - Availability of Pool: {'Yes' if pool else 'No'}

            Analyze the current market trends to determine whether it's a good time to buy, 
            sell, or hold this property. Provide insights and advice for potential buyers or sellers in around 100 words. and give me ans in 100 words only pls 
            it should end with fullstop.
            """

            # Call Cohere API for insights
            response = co.generate(
                model='command-xlarge-nightly',
                prompt=prompt,
                max_tokens=200,  # Ensure the response has enough words for 150 words
                temperature=0.7,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )

            # Extract the AI suggestion
            generated_text = response.generations[0].text.strip()

            # Return as JSON response
            return jsonify({"output": generated_text})
        
        except Exception as e:
            print("Error during house AI suggestion:", str(e))
            return jsonify({"error": f"Error: {str(e)}"})
    
    # For GET request, return an error response
    return jsonify({"error": "Invalid request method."})

if __name__ == '__main__':
    with app.app_context():  # Push the application context
        db.create_all()  # Creates the database tables if they don't exist
    app.run(debug=True)
