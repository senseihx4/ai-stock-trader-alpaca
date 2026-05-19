import os
import numpy as np
import torch
import pickle
import ta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from model import StockLSTM
from stocks import STOCKS

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs("models", exist_ok=True)

def fetch_data(ticker):
    df = yf.download(ticker, start="2022-01-01", end="2026-01-01", progress=False)
    return df

def prepare_data(df, seq_length=30):
    data = pd.DataFrame()
    data['Close'] = df['Close'].squeeze()
    data['Volume'] = df['Volume'].squeeze()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data = data.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i+seq_length])
        y.append(scaled[i+seq_length][0])

    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(device)
    return X, y, scaler

def train_model(ticker):
    print(f"\nTraining {ticker}...")
    df = fetch_data(ticker)

    if len(df) < 60:
        print(f"Not enough data for {ticker}, skipping.")
        return

    X, y, scaler = prepare_data(df)

    model = StockLSTM(input_size=5, hidden_size=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X).squeeze()
        loss = criterion(output, y.squeeze())
        loss.backward()
        optimizer.step()
    
    # Save model AND scaler together
    torch.save(model.state_dict(), f'models/{ticker}_model.pth')
    with open(f'models/{ticker}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ {ticker} saved!")

for ticker in STOCKS:
    try:
        train_model(ticker)
    except Exception as e:
        print(f"❌ {ticker} failed: {e}")

print("\n🎉 All models trained and scalers saved!")