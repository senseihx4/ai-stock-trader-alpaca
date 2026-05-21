"""
One-time bulk download of 10 years of daily OHLCV data.
Run from project root:
    python scripts/download.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from app.core.stocks import STOCKS


def download_all(symbols, period="10y", interval="1d"):
    all_dfs = []

    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] Downloading {symbol}...")
            df = yf.download(symbol, period=period, interval=interval, progress=False)

            if df.empty:
                print(f"⚠️  {symbol} skipped — no data")
                continue

            df.dropna(inplace=True)
            df["Symbol"] = symbol
            df.reset_index(inplace=True)
            all_dfs.append(df)
            print(f"✅ {symbol} ({len(df)} rows)")

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]]
    final_df.to_csv("stocks_10y.csv", index=False)
    print(f"\n✅ Saved → stocks_10y.csv  ({len(final_df):,} rows)")


if __name__ == "__main__":
    download_all(STOCKS)
