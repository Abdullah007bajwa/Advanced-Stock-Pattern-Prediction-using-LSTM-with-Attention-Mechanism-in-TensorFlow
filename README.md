# Advanced Stock Pattern Prediction using LSTM with Attention Mechanism in TensorFlow

## Introduction

In the rapidly evolving world of financial markets, accurate predictions are akin to a holy grail. As we seek more sophisticated techniques to interpret market trends, machine learning emerges as a beacon of hope. Among various machine learning models, Long Short-Term Memory (LSTM) networks have gained significant attention. When combined with the attention mechanism, these models become even more powerful, especially in analyzing time-series data like stock prices. This guide delves into the intriguing world of LSTM networks paired with attention mechanisms, focusing on predicting the pattern of the next four candles in the stock price of various companies, utilizing data from Yahoo Finance (yfinance).

## Table of Contents
1. [Understanding LSTM and Attention in Financial Modelling](#section-1)
2. [Setting Up Your Environment](#section-2)
3. [Data Preprocessing and Preparation](#section-3)
4. [Building the LSTM with Attention Model](#section-4)
5. [Training and Saving Models for Multiple Stocks](#section-5)
6. [Creating a Flask Web Application](#section-6)
7. [Evaluating Model Performance](#section-7)
8. [Predicting the Next 4 Candles](#section-8)

## Section 1: Understanding LSTM and Attention in Financial Modelling <a name="section-1"></a>

### The Basics of LSTM Networks
LSTM networks are a type of Recurrent Neural Network (RNN) designed to remember and process sequences of data over long periods. They preserve information for long durations using three gates: the input, forget, and output gates. These gates manage the flow of information, mitigating the issue of vanishing gradients — a common problem in standard RNNs.

### Attention Mechanism: Enhancing LSTM
The attention mechanism, initially popularized in natural language processing, operates on a simple concept: not all parts of the input sequence are equally important. By allowing the model to focus on specific parts of the input sequence, the attention mechanism enhances the model’s context understanding capabilities.

### The Relevance in Financial Pattern Prediction
Combining LSTM with attention mechanisms creates a robust model for financial pattern prediction. The financial market is influenced by many factors and exhibits non-linear characteristics. LSTM networks, especially when combined with an attention mechanism, adeptly capture these patterns, offering deeper understanding and more accurate forecasts of future stock movements.

## Section 2: Setting Up Your Environment <a name="section-2"></a>

To build our LSTM model with attention for predicting stock patterns, we will use Google Colab. Colab provides a free Jupyter notebook environment with GPU support, ideal for running deep learning models.

### Installing Required Libraries
```python
!pip install tensorflow -qqq
!pip install keras -qqq
!pip install yfinance -qqq
!pip install flask -qqq
```

### Importing Libraries
```python
import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, AdditiveAttention, Permute, Reshape, Multiply, Flatten
from flask import Flask, request, jsonify, render_template
```

## Section 3: Data Preprocessing and Preparation <a name="section-3"></a>

### Fetching Historical Data
```python
def download_stock_data(stock_symbol, period='5y'):
    return yf.download(stock_symbol, period=period)
```

### Data Cleaning and Normalization
```python
def preprocess_data(data):
    closing_prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))
    return scaled_data, scaler
```

### Creating Sequences
```python
def create_sequences(scaled_data):
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)
```

### Train-Test Split and Reshaping Data for LSTM
```python
def reshape_data(X, y):
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y
```

## Section 4: Building the LSTM with Attention Model <a name="section-4"></a>

### Creating the Model
```python
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = LSTM(units=50, return_sequences=True)(x)

    attention = AdditiveAttention(name='attention_weight')
    x_permuted = Permute((2, 1))(x)
    x_reshaped = Reshape((-1, input_shape[0]))(x_permuted)
    attention_result = attention([x_reshaped, x_reshaped])
    multiply_layer = Multiply()([x_reshaped, attention_result])
    x_permuted_back = Permute((2, 1))(multiply_layer)
    x_reshaped_back = Reshape((-1, 50))(x_permuted_back)

    x_flattened = Flatten()(x_reshaped_back)
    x_dense = Dense(1)(x_flattened)
    x_dropout = Dropout(0.2)(x_dense)
    x_batchnorm = BatchNormalization()(x_dropout)

    model = Model(inputs=inputs, outputs=x_batchnorm)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```

## Section 5: Training and Saving Models for Multiple Stocks <a name="section-5"></a>

### Training the Model
```python
def train_model(stock_symbol):
    data = download_stock_data(stock_symbol)
    if len(data) < 60:
        print(f"Not enough data to train model for {stock_symbol}.")
        return None, None

    scaled_data, scaler = preprocess_data(data)
    X, y = create_sequences(scaled_data)
    X_train, y_train = reshape_data(X, y)

    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    return model, scaler
```

### Saving the Model
```python
def save_model(model, scaler, stock_symbol):
    model_dir = f'models/{stock_symbol}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(f'{model_dir}/model.h5')
    np.save(f'{model_dir}/scaler.npy', scaler)
    print(f"Model and scaler saved for {stock_symbol}.")
```

### Training and Saving Models for Multiple Stocks
```python
stock_list = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

for stock in stock_list:
    model, scaler = train_model(stock)
    if model is not None:
        save_model(model, scaler, stock)
```

## Section 6: Creating a Flask Web Application <a name="section-6"></a>

### Flask Application Setup
```python
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
```

## Section 7: Evaluating Model Performance <a name="section-7"></a>

### Evaluating with the Test Set
```python
def evaluate_model

(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
```

### Making Future Predictions
```python
def predict_future(model, scaler, stock_data):
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

    return predictions
```
