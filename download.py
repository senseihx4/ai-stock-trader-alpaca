import yfinance as yf
import pandas as pd
import os

STOCKS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "ORCL",
    "CRM", "ADBE", "NFLX", "PYPL", "QCOM", "TXN", "IBM", "CSCO", "UBER", "LYFT",

    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB",
    "PNC", "TFC", "COF", "MET", "PRU", "AFL", "ALL", "CB", "MMC", "AON",

    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "ISRG", "SYK", "BDX", "ZTS", "VRTX", "REGN", "HUM", "CVS", "CI",

    # Consumer
    "WMT", "HD", "MCD", "SBUX", "NKE", "TGT", "COST", "LOW", "TJX", "DG",
    "DLTR", "KO", "PEP", "PG", "CL", "KMB", "EL", "GIS", "K", "HSY",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "OXY", "HAL",

    # Industrial
    "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "NOC",

    # Telecom & Media
    "T", "VZ", "TMUS", "DIS", "CMCSA", "PARA", "WBD", "FOX", "NYT",
]

STOCKS = list(dict.fromkeys(STOCKS))  # remove duplicates

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
            df["Symbol"] = symbol          # add stock name column
            df.reset_index(inplace=True)   # make Date a column
            all_dfs.append(df)
            print(f"✅ {symbol} ({len(df)} rows)")

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

    # Combine all into one DataFrame
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns
    final_df = final_df[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]]
    
    # Save to single CSV
    final_df.to_csv("stocks_10y.csv", index=False)
    print(f"\n✅ Saved all data → stocks_10y.csv")
    print(f"📊 Total rows: {len(final_df):,}")

if __name__ == "__main__":
    download_all(STOCKS)