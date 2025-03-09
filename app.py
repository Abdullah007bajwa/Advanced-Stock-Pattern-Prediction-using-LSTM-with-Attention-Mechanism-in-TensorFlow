import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS  # Optional: Enable if your frontend is hosted elsewhere

app = Flask(__name__)
CORS(app)  # Remove this line if CORS is not needed

def get_model_and_scaler(stock_symbol):
    model_dir = f'models/{stock_symbol}'
    if not os.path.exists(f'{model_dir}/model.h5') or not os.path.exists(f'{model_dir}/scaler.npy'):
        raise FileNotFoundError(f"Model or scaler not found for stock symbol: {stock_symbol}")
    model = load_model(f'{model_dir}/model.h5')
    scaler = np.load(f'{model_dir}/scaler.npy', allow_pickle=True).item()
    return model, scaler

@app.route('/')
def index():
    app.logger.info('Index page accessed')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        stock_symbol = data.get('stock_symbol')
        input_date = data.get('input_date')

        # Validate inputs
        if not stock_symbol or not input_date:
            return jsonify({'error': 'Missing stock_symbol or input_date'}), 400

        # Load model and scaler
        model, scaler = get_model_and_scaler(stock_symbol)

        # Fetch stock data
        end_date = input_date
        start_date = (pd.to_datetime(input_date) - pd.DateOffset(days=90)).strftime('%Y-%m-%d')
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        if len(stock_data) < 60:
            return jsonify({'error': 'Not enough data to make a prediction.'})

        last_60_days = stock_data['Close'].values[-60:]
        scaled_data = scaler.transform(last_60_days.reshape(-1, 1))

        # Prepare data for prediction
        X_test = []
        X_test.append(scaled_data)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Generate predictions for the next 4 days
        predictions = []
        for _ in range(4):
            prediction = model.predict(X_test)
            predictions.append(scaler.inverse_transform(prediction)[0][0])
            X_test = np.append(X_test[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

        prediction_dates = pd.date_range(start=pd.to_datetime(input_date) + pd.Timedelta(days=1), periods=4)

        return jsonify({
            'predictions': {str(date.date()): float(pred) for date, pred in zip(prediction_dates, predictions)}
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
