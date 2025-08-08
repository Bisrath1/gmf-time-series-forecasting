import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def train_test_split(prices, split_date='2023-12-31'):
    train = prices[:split_date]
    test = prices[split_date:]
    return train, test

def evaluate_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mae, rmse, mape

def arima_forecast(train, test):
    model = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(train)
    n_periods = len(test)
    forecast = model.predict(n_periods=n_periods)
    return forecast

def create_lstm_dataset(series, look_back=60):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def lstm_forecast(train, test, epochs=20, batch_size=32):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1,1))
    test_scaled = scaler.transform(test.values.reshape(-1,1))
    
    X_train, y_train = create_lstm_dataset(train_scaled)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Forecast
    inputs = train_scaled[-60:].reshape(1, 60, 1)
    preds_scaled = []
    for _ in range(len(test)):
        pred = model.predict(inputs)[0,0]
        preds_scaled.append(pred)
        inputs = np.append(inputs[:,1:,:], [[[pred]]], axis=1)
    
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    return preds

def plot_forecasts(train, test, arima_pred, lstm_pred):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, arima_pred, label='ARIMA Forecast')
    plt.plot(test.index, lstm_pred, label='LSTM Forecast')
    plt.legend()
    plt.title('TSLA Price Forecast Comparison')
    plt.show()

if __name__ == "__main__":
    prices = pd.read_csv('data/processed/prices.csv', index_col=0, parse_dates=True)
    tsla = prices['TSLA']
    train, test = train_test_split(tsla)
    
    print("Training ARIMA...")
    arima_pred = arima_forecast(train, test)
    
    print("Training LSTM...")
    lstm_pred = lstm_forecast(train, test)
    
    mae_a, rmse_a, mape_a = evaluate_metrics(test, arima_pred)
    mae_l, rmse_l, mape_l = evaluate_metrics(test, lstm_pred)
    
    print(f"ARIMA MAE: {mae_a:.4f}, RMSE: {rmse_a:.4f}, MAPE: {mape_a:.2f}%")
    print(f"LSTM MAE: {mae_l:.4f}, RMSE: {rmse_l:.4f}, MAPE: {mape_l:.2f}%")
    
    plot_forecasts(train, test, arima_pred, lstm_pred)
