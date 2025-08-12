# src/gmf_ts/portfolio_from_lstm.py
# Practical portfolio optimization using LSTM forecasts only

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

# Optional (used if we need to produce a quick LSTM forecast fallback)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# PyPortfolioOpt
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

# -----------------------------
# Config
# -----------------------------
PRICES_PATH = "data/processed/prices.csv"
RETURNS_PATH = "data/processed/returns.csv"      # optional (we compute if absent)
LSTM_PREDS_PATH = "reports/lstm_backtest/lstm_backtest_preds.csv"  # expected existing predictions
OUT_DIR = Path("reports/portfolio")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["TSLA", "BND", "SPY"]
TRAIN_END = "2023-12-31"               # training window end (no leakage)
BACKTEST_START = "2024-08-01"          # per your plan
BACKTEST_END = "2025-07-31"
RISK_FREE_RATE = 0.0                   # set to >0 if you want (e.g., 0.02 for 2%)

# -----------------------------
# Utilities
# -----------------------------
def load_prices():
    if not os.path.exists(PRICES_PATH):
        raise FileNotFoundError(f"{PRICES_PATH} not found.")
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    prices = prices[TICKERS].dropna().sort_index()
    return prices

def compute_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns

def load_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(RETURNS_PATH):
        rets = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
        return rets[TICKERS].dropna().sort_index()
    return compute_returns_from_prices(prices)

def try_load_lstm_preds():
    """Try to read your existing LSTM predictions file.
       Expected columns: a date column + one of ['pred','y_pred','lstm_pred'].
       Returns None if not found or invalid.
    """
    if not os.path.exists(LSTM_PREDS_PATH):
        return None

    df = pd.read_csv(LSTM_PREDS_PATH)
    # Try to find a date column
    date_col = None
    for c in df.columns:
        if c.lower() in ["date", "ds", "timestamp"]:
            date_col = c
            break
    if date_col is None:
        # assume first column is date
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # try to find prediction column
    pred_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ["pred", "y_pred", "lstm_pred", "prediction", "forecast"]:
            pred_col = c
            break
    if pred_col is None:
        # if your file is a 2-col file, assume second col is pred
        if df.shape[1] >= 2:
            pred_col = df.columns[1]
        else:
            return None

    preds = df[[date_col, pred_col]].rename(columns={date_col: "Date", pred_col: "TSLA"})
    preds = preds.set_index("Date").sort_index()
    preds = preds[~preds.index.duplicated(keep="last")]
    return preds

def quick_lstm_forecast(prices: pd.Series, start_date: str, end_date: str,
                        look_back=60, epochs=20, batch_size=32, units=64) -> pd.Series:
    """Very small fallback LSTM to produce out-of-sample daily price forecasts if you don't have saved preds."""
    # train on data up to TRAIN_END and forecast [start_date, end_date]
    y = prices.loc[:TRAIN_END].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)

    X, t = [], []
    for i in range(look_back, len(y_scaled)):
        X.append(y_scaled[i - look_back:i, 0])
        t.append(y_scaled[i, 0])
    X = np.array(X).reshape(-1, look_back, 1)
    t = np.array(t)

    model = Sequential([
        LSTM(units, input_shape=(look_back,1), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, t, epochs=epochs, batch_size=batch_size, verbose=0)

    # rolling forecast daily
    horizon_idx = prices.loc[start_date:end_date].index
    last_window = y_scaled[-look_back:].reshape(1, look_back, 1)
    preds_scaled = []
    for _ in range(len(horizon_idx)):
        pred_scaled = model.predict(last_window, verbose=0)[0,0]
        preds_scaled.append(pred_scaled)
        # slide window
        last_window = np.append(last_window[:,1:,:], [[[pred_scaled]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    pred_series = pd.Series(preds, index=horizon_idx, name="TSLA")
    return pred_series

def annualize_return(daily_series: pd.Series) -> float:
    return float(daily_series.mean() * 252)

def annualize_vol(daily_series: pd.Series) -> float:
    return float(daily_series.std(ddof=1) * np.sqrt(252))

def sharpe_ratio(daily_series: pd.Series, rf=0.0) -> float:
    mu = annualize_return(daily_series)
    vol = annualize_vol(daily_series)
    if vol == 0:
        return np.nan
    return (mu - rf) / vol

def max_drawdown(daily_series: pd.Series) -> float:
    cum = (1 + daily_series).cumprod()
    dd = cum / cum.cummax() - 1
    return float(dd.min())

def metrics_block(daily_series: pd.Series, name: str, rf=0.0) -> dict:
    return {
        "name": name,
        "ann_return": annualize_return(daily_series),
        "ann_vol": annualize_vol(daily_series),
        "sharpe": sharpe_ratio(daily_series, rf),
        "max_drawdown": max_drawdown(daily_series)
    }

# -----------------------------
# Core steps
# -----------------------------
def main():
    prices = load_prices()
    returns = load_returns(prices)

    # Split windows
    train_mask = (returns.index <= pd.to_datetime(TRAIN_END))
    backtest_mask = (returns.index >= pd.to_datetime(BACKTEST_START)) & (returns.index <= pd.to_datetime(BACKTEST_END))

    train_returns = returns.loc[train_mask, TICKERS]
    backtest_returns = returns.loc[backtest_mask, TICKERS]
    backtest_prices = prices.loc[backtest_mask, TICKERS]

    # 1) TSLA expected return from LSTM forecasts
    lstm_preds = try_load_lstm_preds()
    if lstm_preds is not None:
        # restrict to backtest window and align to trading days
        tsla_pred_prices = lstm_preds.reindex(backtest_prices.index).ffill().bfill().dropna()
        if tsla_pred_prices.empty:
            # fallback if dates didn't align
            tsla_pred_prices = quick_lstm_forecast(prices["TSLA"], BACKTEST_START, BACKTEST_END)
    else:
        # fallback small LSTM if no file exists
        tsla_pred_prices = quick_lstm_forecast(prices["TSLA"], BACKTEST_START, BACKTEST_END)

    # Convert TSLA price forecast to daily returns
    tsla_forecast_returns = tsla_pred_prices["TSLA"].pct_change().dropna()

    # Expected returns (annualized)
    tsla_mu = annualize_return(tsla_forecast_returns)
    # Use historical mean daily returns (train) for BND & SPY (annualized)
    bnd_mu = annualize_return(train_returns["BND"])
    spy_mu = annualize_return(train_returns["SPY"])

    mu_vec = np.array([tsla_mu, bnd_mu, spy_mu])

    # 2) Covariance matrix S from historical daily returns (training window)
    S = (train_returns[TICKERS].cov()) * 252.0

    # 3) Efficient Frontier portfolios
    ef = EfficientFrontier(mu_vec, S)
    ef.add_objective(objective_functions.L2_reg, gamma=0.001)   # tiny regularization for stability
    w_max_sharpe = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    perf_max_sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE)
    weights_max_sharpe = pd.Series(w_max_sharpe, index=TICKERS)

    ef_min = EfficientFrontier(mu_vec, S)
    ef_min.add_objective(objective_functions.L2_reg, gamma=0.001)
    w_min_vol = ef_min.min_volatility()
    perf_min_vol = ef_min.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE)
    weights_min_vol = pd.Series(w_min_vol, index=TICKERS)

    # Save weights
    weights_df = pd.DataFrame({
        "MaxSharpe": weights_max_sharpe.round(6),
        "MinVol": weights_min_vol.round(6)
    })
    weights_df.to_csv(OUT_DIR / "portfolio_weights.csv")
    print("✅ Saved weights -> reports/portfolio/portfolio_weights.csv")

    # 4) Backtest vs 60/40 SPY/BND
    if backtest_returns.empty:
        raise RuntimeError("Backtest window has no returns; check BACKTEST_START/BACKTEST_END and data coverage.")

    # Strategy returns with fixed weights (no rebalancing within window)
    strat_max_sharpe = (backtest_returns[TICKERS] * weights_max_sharpe.values).sum(axis=1)
    strat_min_vol = (backtest_returns[TICKERS] * weights_min_vol.values).sum(axis=1)

    # Benchmark: 60/40 SPY/BND (ignore TSLA)
    bench_weights = pd.Series({"SPY": 0.6, "BND": 0.4})
    bench_returns = (backtest_returns[["SPY","BND"]] * bench_weights.values).sum(axis=1)

    # Metrics
    metrics = [
        metrics_block(strat_max_sharpe, "Strategy_MaxSharpe", RISK_FREE_RATE),
        metrics_block(strat_min_vol, "Strategy_MinVol", RISK_FREE_RATE),
        metrics_block(bench_returns, "Benchmark_60_40", RISK_FREE_RATE),
    ]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DIR / "backtest_metrics.csv", index=False)
    print("✅ Saved backtest metrics -> reports/portfolio/backtest_metrics.csv")
    print(metrics_df.to_string(index=False))

    # 5) Plots
    # Cumulative returns
    plt.figure(figsize=(11,6))
    (1 + strat_max_sharpe).cumprod().plot(label="Strategy MaxSharpe")
    (1 + strat_min_vol).cumprod().plot(label="Strategy MinVol")
    (1 + bench_returns).cumprod().plot(label="Benchmark 60/40")
    plt.title("Cumulative Returns (Backtest)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cumulative_returns.png", dpi=150)
    plt.close()
    print("✅ Saved cumulative plot -> reports/portfolio/cumulative_returns.png")

    # Efficient frontier scatter (approx) — optional: show weight points only
    plt.figure(figsize=(8,6))
    # scatter the two portfolios
    plt.scatter(perf_max_sharpe[1], perf_max_sharpe[0], marker="o", s=80, label="Max Sharpe")
    plt.scatter(perf_min_vol[1], perf_min_vol[0], marker="^", s=80, label="Min Vol")
    plt.xlabel("Annual Volatility")
    plt.ylabel("Annual Return")
    plt.title("Selected Portfolios")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "selected_portfolios.png", dpi=150)
    plt.close()
    print("✅ Saved selected portfolios scatter -> reports/portfolio/selected_portfolios.png")

    # Save the mu & covariance used (for report reproducibility)
    pd.Series(mu_vec, index=TICKERS, name="annual_mu").to_csv(OUT_DIR / "mu_vector.csv")
    S.to_csv(OUT_DIR / "cov_matrix.csv")
    print("✅ Saved mu_vector.csv and cov_matrix.csv")

    # Also store the TSLA forecast used (so report can cite it)
    tsla_pred_prices.to_csv(OUT_DIR / "tsla_lstm_forecast_prices_used.csv")
    print("✅ Saved tsla_lstm_forecast_prices_used.csv")

    print("\n🎯 Done. Artifacts in reports/portfolio/:")
    for f in OUT_DIR.glob("*"):
        print(" -", f.name)

if __name__ == "__main__":
    main()
