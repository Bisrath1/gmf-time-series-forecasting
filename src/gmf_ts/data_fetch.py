import yfinance as yf
import pandas as pd
from pathlib import Path
import time

def fetch_prices(
    tickers=['TSLA', 'BND', 'SPY'],
    start='2015-07-01',
    end='2025-07-31',
    retries=3,
    delay=3,
    output_path=Path("data/processed/prices.csv")
):
    prices_dict = {}

    for t in tickers:
        attempt = 0
        while attempt < retries:
            try:
                print(f"📥 Fetching {t} (attempt {attempt+1})...")
                df = yf.download(
                    t, start=start, end=end,
                    progress=False, auto_adjust=False
                )
                if not df.empty and 'Adj Close' in df.columns:
                    prices_dict[t] = df['Adj Close'].copy()
                    print(f"✅ {t} fetched ({len(df)} rows).")
                    break
                else:
                    print(f"⚠️ {t} returned empty or missing 'Adj Close'.")
            except Exception as e:
                print(f"❌ Error fetching {t}: {e}")
            attempt += 1
            time.sleep(delay)

    if not prices_dict:
        raise RuntimeError("❌ No data fetched — check network or ticker symbols.")

    # Build DataFrame the safe way
    prices = pd.concat(prices_dict, axis=1)
    prices.columns = list(prices_dict.keys())
    prices = prices.sort_index().ffill().dropna()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path)

    print(f"💾 Data saved to {output_path.resolve()}")
    return prices

if __name__ == "__main__":
    df = fetch_prices()
    print(df.head())
