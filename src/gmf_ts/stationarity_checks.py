# src/gmf_ts/stationarity_checks.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

prices = pd.read_csv('data/processed/prices.csv', index_col=0, parse_dates=True)
tsla = prices['TSLA'].dropna()

def adf_report(series, name='series'):
    res = adfuller(series, autolag='AIC')
    adf_stat, pvalue, usedlag, nobs, icbest, critical_values = res
    print(f"ADF test for {name}:")
    print(f"  ADF statistic: {adf_stat:.4f}")
    print(f"  p-value: {pvalue:.4f}")
    print("  critical values:")
    for k, v in critical_values.items():
        print(f"    {k}: {v:.4f}")
    return res

# ADF on raw price
adf_report(tsla, 'TSLA price')

# ADF on log price
log_tsla = np.log(tsla)
adf_report(log_tsla.dropna(), 'log(TSLA)')

# ADF on returns
returns = tsla.pct_change().dropna()
adf_report(returns, 'TSLA returns')
