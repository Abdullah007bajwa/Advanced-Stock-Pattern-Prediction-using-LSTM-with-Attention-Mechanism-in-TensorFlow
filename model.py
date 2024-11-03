import yfinance as yf
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, AdditiveAttention, Permute, Reshape, Multiply, Flatten

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

def train_model(stock_symbol):
    data = yf.download(stock_symbol, period='5y')
    if len(data) < 60:
        print(f"Not enough data to train model for {stock_symbol}.")
        return None, None

    closing_prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X), np.array(y)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    return model, scaler

def save_model(model, scaler, stock_symbol):
    model_dir = f'models/{stock_symbol}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(f'{model_dir}/model.h5')
    np.save(f'{model_dir}/scaler.npy', scaler)
    print(f"Model and scaler saved for {stock_symbol}.")

stock_list = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

for stock in stock_list:
    model, scaler = train_model(stock)
    if model is not None:
        save_model(model, scaler, stock)