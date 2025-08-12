# src/gmf_ts/preprocess_stationary.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os

DATA_PATH = 'data/processed/prices.csv'
OUTPUT_PATH = 'data/processed/stationary_prices.csv'

def adf_test(series, name='Series'):
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, pvalue, usedlag, nobs, critical_values, icbest = result  # ✅ correct order
    print(f"ADF test for {name}:")
    print(f"  ADF statistic: {adf_stat:.4f}")
    print(f"  p-value: {pvalue:.4f}")
    print("  Critical values:")
    for k, v in critical_values.items():
        print(f"    {k}%: {v:.4f}")
    print(f"  Stationary: {'Yes' if pvalue < 0.05 else 'No'}\n")
    return pvalue, pvalue < 0.05



    """
    Run ADF test and return p-value + stationarity flag.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, pvalue, _, _, _, critical_values = result
    print(f"ADF test for {name}:")
    print(f"  ADF statistic: {adf_stat:.4f}")
    print(f"  p-value: {pvalue:.4f}")
    for k, v in critical_values.items():
        print(f"    {k}: {v:.4f}")
    print(f"  Stationary: {'Yes' if pvalue < 0.05 else 'No'}\n")
    return pvalue, pvalue < 0.05

def make_stationary(series):
    """
    Try raw, log, diff, and logdiff transformations to get stationarity.
    Returns transformed series and description.
    """
    checks = [
        ('raw', series),
        ('log', np.log(series)),
        ('diff', series.diff()),
        ('logdiff', np.log(series).diff())
    ]

    for name, transformed in checks:
        pvalue, stationary = adf_test(transformed, name)
        if stationary:
            return transformed.dropna(), name
    raise ValueError("No stationary transformation found.")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    stationary_data = {}

    for col in prices.columns:
        series = prices[col].dropna()
        print(f"Checking stationarity for {col}...\n")
        transformed, method = make_stationary(series)
        stationary_data[col] = transformed
        print(f"✅ {col} - Using '{method}' transformation for modeling.\n")

    stationary_df = pd.DataFrame(stationary_data)
    stationary_df.to_csv(OUTPUT_PATH)
    print(f"📁 Stationary dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

