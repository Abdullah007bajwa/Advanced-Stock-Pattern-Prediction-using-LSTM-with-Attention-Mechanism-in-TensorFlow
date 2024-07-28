import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def get_model_and_scaler(stock_symbol):
    model_dir = f'models/{stock_symbol}'
    model = load_model(f'{model_dir}/model.h5')
    scaler = np.load(f'{model_dir}/scaler.npy', allow_pickle=True).item()
    return model, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stock_symbol = data['stock_symbol']
    input_date = data['input_date']

    model, scaler = get_model_and_scaler(stock_symbol)

    end_date = input_date
    start_date = (pd.to_datetime(input_date) - pd.DateOffset(days=90)).strftime('%Y-%m-%d')
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    if len(stock_data) < 60:
        return jsonify({'error': 'Not enough data to make a prediction.'})

    last_60_days = stock_data['Close'].values[-60:]
    scaled_data = scaler.transform(last_60_days.reshape(-1, 1))

    X_test = []
    X_test.append(scaled_data)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = []
    for _ in range(4):
        prediction = model.predict(X_test)
        predictions.append(scaler.inverse_transform(prediction)[0][0])
        X_test = np.append(X_test[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    prediction_dates = pd.date_range(start=pd.to_datetime(input_date) + pd.Timedelta(days=1), periods=4)

    return jsonify({'predictions': {str(date.date()): pred for date, pred in zip(prediction_dates, predictions)}})

if __name__ == '__main__':
    app.run(debug=True)
