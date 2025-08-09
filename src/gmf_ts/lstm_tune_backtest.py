# src/gmf_ts/lstm_tune_backtest.py
import os
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# Utilities
# -----------------------
def create_lstm_dataset(series, look_back):
    X, y = [], []
    arr = series.reshape(-1, 1)
    for i in range(len(arr) - look_back):
        X.append(arr[i : i + look_back].flatten())
        y.append(arr[i + look_back, 0])
    X = np.array(X)
    y = np.array(y)
    return X, y

def build_lstm_model(input_shape, units=50, dropout=0.0, lr=None):
    # lr not used here but kept for compatibility
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, activation='tanh'))
    if dropout and dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def metrics(true, pred):
    true = np.array(true, dtype=float)
    pred = np.array(pred, dtype=float)
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    # avoid division by zero
    mask = true != 0
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = (np.abs((true[mask] - pred[mask]) / true[mask]).mean()) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# -----------------------
# Tuning (simple grid)
# -----------------------
def tune_lstm(train_series, look_back=60, param_grid=None, epochs=50, batch_size=32, val_frac=0.1, patience=5):
    if param_grid is None:
        param_grid = {
            'units': [32, 50],
            'dropout': [0.0, 0.2],
            'epochs': [20],   # keep small for tuning pass; increase later
            'batch_size': [32]
        }

    combos = list(itertools.product(*(param_grid[k] for k in param_grid)))
    keys = list(param_grid.keys())

    # prepare scaler + data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()

    # split train/val (chronological)
    n_val = int(len(scaled) * val_frac)
    train_scaled = scaled[:-n_val] if n_val > 0 else scaled
    val_scaled = scaled[-n_val:] if n_val > 0 else np.array([])

    best = None
    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        # build dataset from train_scaled
        X_train, y_train = create_lstm_dataset(train_scaled, look_back)
        if len(X_train) == 0:
            continue
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        # validation
        if n_val > 0:
            # need some points from train end + val to build sequences (use tail of train for continuity)
            combined_for_val = np.concatenate([train_scaled[-look_back:], val_scaled])
            X_val, y_val = create_lstm_dataset(combined_for_val, look_back)
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        else:
            X_val, y_val = None, None

        model = build_lstm_model((look_back, 1), units=params['units'], dropout=params['dropout'])
        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True) if X_val is not None else EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=params.get('epochs', epochs), batch_size=params.get('batch_size', batch_size),
                  validation_data=(X_val, y_val) if X_val is not None else None, verbose=0, callbacks=[es])

        # evaluate on validation if available else training
        if X_val is not None and len(X_val) > 0:
            pred_val = model.predict(X_val, verbose=0).flatten()
            # inverse scale
            pred_val = scaler.inverse_transform(pred_val.reshape(-1,1)).flatten()
            y_val_true = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
            m = metrics(y_val_true, pred_val)
        else:
            pred_train = model.predict(X_train, verbose=0).flatten()
            pred_train = scaler.inverse_transform(pred_train.reshape(-1,1)).flatten()
            y_train_true = scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
            m = metrics(y_train_true, pred_train)

        results.append({'params': params, 'metrics': m})
        if best is None or m['RMSE'] < best['metrics']['RMSE']:
            best = {'params': params, 'metrics': m}

    return best, results, scaler

# -----------------------
# Rolling backtest (retrain_every to speed up)
# -----------------------
def rolling_backtest_lstm(train_series, test_series, best_params, look_back=60,
                          retrain_every=30, epochs=50, batch_size=32, lstm_units=50, dropout=0.0, save_dir='reports'):
    os.makedirs(save_dir, exist_ok=True)
    all_preds = []
    all_dates = list(test_series.index)
    scaler = MinMaxScaler()
    combined = pd.concat([train_series, test_series])
    scaled_full = scaler.fit_transform(combined.values.reshape(-1,1)).flatten()

    model = None
    last_trained_until = len(train_series) - 1  # index in combined used for training end
    # initial training on full train set
    train_scaled = scaled_full[: len(train_series) ]
    if len(train_scaled) <= look_back:
        raise ValueError("Train length <= look_back, increase data or reduce look_back")

    model = build_lstm_model((look_back, 1), units=lstm_units, dropout=dropout)
    X_train, y_train = create_lstm_dataset(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

    # prepare rolling input sequence from train tail
    input_seq = train_scaled[-look_back:].reshape(1, look_back, 1)

    for t in range(len(test_series)):
        # Optionally retrain every `retrain_every` steps
        if t % retrain_every == 0 and t > 0:
            # new training window: train + seen test up to t
            train_window_len = len(train_series) + t
            train_scaled = scaled_full[:train_window_len]
            X_train, y_train = create_lstm_dataset(train_scaled, look_back)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            model = build_lstm_model((look_back, 1), units=lstm_units, dropout=dropout)
            es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
            input_seq = train_scaled[-look_back:].reshape(1, look_back, 1)

        pred_scaled = model.predict(input_seq, verbose=0)[0,0]
        all_preds.append(pred_scaled)
        # append predicted value to input_seq for next step
        input_seq = np.append(input_seq[:,1:,:], [[[pred_scaled]]], axis=1)

    # inverse transform preds and align with test index
    preds = scaler.inverse_transform(np.array(all_preds).reshape(-1,1)).flatten()
    preds_series = pd.Series(preds, index=test_series.index)

    # compute residuals & metrics
    res = test_series.values - preds
    res_series = pd.Series(res, index=test_series.index)
    m = metrics(test_series.values, preds)

    # Save outputs
    out_dir = Path(save_dir)
    preds_series.to_csv(out_dir / 'lstm_backtest_preds.csv', header=['pred'])
    res_series.to_csv(out_dir / 'lstm_backtest_residuals.csv', header=['residual'])
    pd.DataFrame([m]).to_csv(out_dir / 'lstm_backtest_metrics.csv', index=False)

    # Plot forecast vs actual
    plt.figure(figsize=(12,6))
    plt.plot(train_series.index, train_series.values, label='Train')
    plt.plot(test_series.index, test_series.values, label='Test')
    plt.plot(preds_series.index, preds_series.values, label='LSTM Rolling Pred')
    plt.legend()
    plt.title('LSTM Rolling Backtest Forecast')
    plt.savefig(out_dir / 'lstm_backtest_forecast.png')
    plt.close()

    # Plot residuals time series
    plt.figure(figsize=(12,4))
    plt.plot(res_series.index, res_series.values)
    plt.title('LSTM Backtest Residuals (Actual - Pred)')
    plt.savefig(out_dir / 'lstm_backtest_residuals_ts.png')
    plt.close()

    # Plot residuals histogram
    plt.figure(figsize=(6,4))
    plt.hist(res_series.values, bins=40)
    plt.title('Residuals Histogram')
    plt.savefig(out_dir / 'lstm_backtest_residuals_hist.png')
    plt.close()

    print("LSTM backtest finished. Metrics:", m)
    print("Saved preds, residuals, metrics and plots to", out_dir.resolve())
    return preds_series, res_series, m

# -----------------------
# Main runner
# -----------------------
if __name__ == "__main__":
    # Load prices
    prices = pd.read_csv('data/processed/prices.csv', index_col=0, parse_dates=True)
    tsla = prices['TSLA'].sort_index()

    # Train/test split
    split_date = '2023-12-31'
    train = tsla[:split_date]
    test = tsla[split_date:]

    # Tuning grid (small, expand if you want)
    param_grid = {
        'units': [32, 50],
        'dropout': [0.0, 0.2],
        'epochs': [20],
        'batch_size': [32]
    }
    print("Tuning LSTM (small grid, fast)...")
    best, results, scaler_for_tuning = tune_lstm(train, look_back=60, param_grid=param_grid, epochs=20, batch_size=32, val_frac=0.1)
    print("Best tuning result:", best)

    # Use best params (or fallback)
    bp = best['params'] if best else {'units':50, 'dropout':0.0, 'epochs':20, 'batch_size':32}
    units = bp.get('units', 50)
    dropout = bp.get('dropout', 0.0)
    epochs = bp.get('epochs', 20)
    batch_size = bp.get('batch_size', 32)
    look_back = 60

    # Rolling backtest (retrain every N steps to save time)
    preds_series, res_series, metrics_dict = rolling_backtest_lstm(train, test,
                                                                  best_params=bp,
                                                                  look_back=look_back,
                                                                  retrain_every=30,
                                                                  epochs=epochs,
                                                                  batch_size=batch_size,
                                                                  lstm_units=units,
                                                                  dropout=dropout,
                                                                  save_dir='reports/lstm_backtest')

    # print a short summary
    print("Overall LSTM backtest metrics:", metrics_dict)

    # Save a summary CSV of tuning results
    summary_rows = []
    for r in results:
        row = r['params'].copy()
        row.update(r['metrics'])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv('reports/lstm_tuning_summary.csv', index=False)
    print("Tuning summary saved to reports/lstm_tuning_summary.csv")
