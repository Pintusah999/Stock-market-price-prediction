from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import os
import base64
from tensorflow.keras.models import load_model
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__)

def load_model_and_scaler():
    # Update paths to point to the models folder
    model = load_model(os.path.join('models', 'lstm_stock_price_model.h5'))  # Update with path to model
    with open(os.path.join('models', 'scaler.pkl'), 'rb') as f:  # Update with path to scaler
        scaler = pickle.load(f)
    return model, scaler

# Fetch stock data
def fetch_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data[['Close']]

# Prepare data for prediction
def prepare_data_for_prediction(data, time_step=60):
    model, scaler = load_model_and_scaler()
    scaled_data = scaler.transform(data[['Close']].values)
    X = []
    X.append(scaled_data[-time_step:].reshape(-1, 1))
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

# Predict future prices
def predict_future_prices(model, data, days=30):
    X, scaler = prepare_data_for_prediction(data)
    future_prices = []
    for _ in range(days):
        predicted_price = model.predict(X)
        future_prices.append(predicted_price[0][0])
        X = np.append(X[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices.flatten()

# Generate base64 images for actual and predicted graphs
def generate_graphs(data, predicted_prices, ticker):
    # Actual Price Graph (Plotly)
    fig_actual = go.Figure()
    fig_actual.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price'))
    fig_actual.update_layout(
        title=f'Actual Stock Price for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )
    img_actual = fig_actual.to_html(full_html=False)

    # Predicted Price Graph (Plotly)
    fig_predicted = go.Figure()
    # Use day numbers (1, 2, 3, ..., 30) as the x-axis for predicted prices
    fig_predicted.add_trace(go.Scatter(x=list(range(1, 31)), y=predicted_prices, mode='lines', name='Predicted Price'))
    fig_predicted.update_layout(
        title=f'Predicted Stock Price for {ticker}',
        xaxis_title='Days (1-30)',
        yaxis_title='Price',
        xaxis=dict(tickmode='array', tickvals=list(range(1, 31))),  # This will display the days 1 to 30 on the x-axis
        template='plotly_dark'
    )
    img_predicted = fig_predicted.to_html(full_html=False)

    return img_actual, img_predicted

@app.route('/', methods=['GET', 'POST'])
def index():
    ticker = request.args.get('ticker')
    if ticker:
        # Fetch the stock data
        data = fetch_stock_data(ticker)
        
        model, scaler = load_model_and_scaler()
        predicted_prices = predict_future_prices(model, data)

        # Generate the graphs
        img_base64_actual, img_base64_predicted = generate_graphs(data, predicted_prices, ticker)

        # Create a table for the predicted prices (for the next 30 days)
        predicted_data = [(f"Day {i+1}", price) for i, price in enumerate(predicted_prices)]

        return render_template('index.html', 
                               ticker=ticker, 
                               img_base64_actual=img_base64_actual, 
                               img_base64_predicted=img_base64_predicted,
                               predicted_data=predicted_data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)