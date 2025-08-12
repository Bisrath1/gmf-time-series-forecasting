import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ===============================
# Load stationary data correctly
# ===============================
df = pd.read_csv(
    "data/processed/stationary_prices.csv",
    index_col=0,
    parse_dates=True
)

def run_arima(series, ticker):
    print(f"\n🔹 Running ARIMA for {ticker}...")
    
    # Drop NaNs just in case
    series = series.dropna()

    # Train/test split
    train_size = int(len(series) * 0.8)
    train, test = series.iloc[:train_size], series.iloc[train_size:]

    if len(test) == 0:
        print(f"⚠️ No test data for {ticker} — skipping.")
        return None

    try:
        # Fit ARIMA model
        model = auto_arima(
            train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
        print(f"Best ARIMA order for {ticker}: {model.order}")

        # Forecast same length as test set
        forecast_values = model.predict(n_periods=len(test))
        forecast = pd.Series(forecast_values, index=test.index)

        # Evaluate
        mae = mean_absolute_error(test, forecast)
        rmse = mean_squared_error(test, forecast, squared=False)
        print(f"📊 {ticker} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return forecast

    except Exception as e:
        print(f"⚠️ Error for {ticker}: {e}")
        return None

# ===============================
# Run ARIMA for all tickers
# ===============================
forecasts = {}
for ticker in df.columns:
    forecasts[ticker] = run_arima(df[ticker], ticker)

# ===============================
# Save only non-empty forecasts
# ===============================
valid_forecasts = {t: fc for t, fc in forecasts.items() if fc is not None}

if valid_forecasts:
    forecast_df = pd.DataFrame(valid_forecasts)
    forecast_df.to_csv("data/processed/arima_forecasts.csv")
    print("\n💾 Forecasts saved to data/processed/arima_forecasts.csv")
else:
    print("\n⚠️ No valid forecasts generated.")
