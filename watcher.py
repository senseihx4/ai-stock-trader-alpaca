import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from model import StockLSTM
from trader import buy_stock, sell_stock, is_market_open
from stocks import STOCKS
import time
import schedule
import pickle

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def fetch_data(ticker, start="2022-01-01", end="2026-01-01"):
    df = yf.download(ticker, start=start, end=end)
    return df


def prepare_data(df, seq_length=30):
    data = df[['Close']].copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    

    last_sequence = scaled[-seq_length:]
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(device)
    
    current_price = float(data['Close'].to_numpy().flatten()[-1])

    return last_sequence, scaler, current_price

def predict_next_day(ticker):
    import pandas as pd
    import ta

    model = StockLSTM(input_size=5, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(f'models/{ticker}_model.pth', map_location=device))
    model.to(device)
    model.eval()

    with open(f'models/{ticker}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    df = fetch_data(ticker)
    df.columns = df.columns.get_level_values(0)

    # Build features in EXACT same order as train_all.py
    data = pd.DataFrame()
    data['Close']  = df['Close'].squeeze()
    data['Volume'] = df['Volume'].squeeze()
    data['RSI']    = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD']   = ta.trend.MACD(data['Close']).macd()
    data['MA20']   = data['Close'].rolling(window=20).mean()
    data = data.dropna()

    current_price = float(data['Close'].values[-1])

    scaled = scaler.transform(data)
    last_sequence = scaled[-30:]
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(last_sequence.unsqueeze(0)).squeeze().cpu().numpy()

    # Inverse transform — Close is index 0
    dummy = pd.DataFrame(
        [[float(pred), 0.0, 0.0, 0.0, 0.0]],
        columns=['Close', 'Volume', 'RSI', 'MACD', 'MA20']
    )
    predicted_price = float(scaler.inverse_transform(dummy)[0][0])

    if predicted_price > current_price:
        return "BUY", current_price, predicted_price
    else:
        return "SELL", current_price, predicted_price




buy_counter = {ticker: 0 for ticker in STOCKS}
buy_prices = {}
MAX_BUYS = 5
MIN_GAIN_PERCENT = 1.5

def run_set():
    if not is_market_open():
        print("Market is closed. Skipping.")
        return

    for ticker in STOCKS:
        try:
            signal, current, predicted = predict_next_day(ticker)
            gain_percent = ((predicted - current) / current) * 100

            if ticker in buy_prices:
                profit = ((current - buy_prices[ticker]) / buy_prices[ticker]) * 100
                if profit >= MIN_GAIN_PERCENT:
                    sell_stock(ticker, 1)
                    print(f"💰 PROFIT SELL {ticker} | Bought: ${buy_prices[ticker]:.2f} | Now: ${current:.2f} | Profit: {profit:.2f}%")
                    del buy_prices[ticker]
                    buy_counter[ticker] = 0
                    continue

            if signal == "BUY" and gain_percent >= MIN_GAIN_PERCENT and buy_counter[ticker] < MAX_BUYS:
                buy_stock(ticker, 1)
                buy_prices[ticker] = current
                buy_counter[ticker] += 1
                print(f"✅ BUY {ticker} | Current: ${current:.2f} | Predicted: ${predicted:.2f} | Gain: {gain_percent:.2f}%")
            else:
                direction = "📈 UP" if signal == "BUY" else "📉 DOWN"
                print(f"{ticker} | ${current:.2f} | {direction} {gain_percent:.2f}% | No trade")

        except Exception as e:
            print(f"❌ {ticker} error: {e}")






schedule.every(10).minutes.do(run_set)

print("🤖 Trading bot started!")
run_set()  # run immediately on start

while True:
    schedule.run_pending()
    time.sleep(1)