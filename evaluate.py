import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import pandas as pd
from model import StockLSTM
from sklearn.metrics import mean_squared_error

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def evaluate_stock(ticker):
    # Load model and scaler
    model = StockLSTM(input_size=5, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(f'models/{ticker}_model.pth', map_location=device))
    model.to(device)
    model.eval()

    with open(f'models/{ticker}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Fetch and prepare data
    df = yf.download(ticker, start="2024-01-01", end="2026-01-01", progress=False)
    data = pd.DataFrame()
    data['Close'] = df['Close'].squeeze()
    data['Volume'] = df['Volume'].squeeze()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data = data.dropna()

    scaled = scaler.transform(data)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled) - 30):
        X.append(scaled[i:i+30])
        y.append(scaled[i+30][0])

    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = np.array(y)

    # Predict
    with torch.no_grad():
        preds = model(X).squeeze().cpu().numpy()

    # Inverse transform
    dummy = np.zeros((len(preds), 5))
    dummy[:, 0] = preds
    predicted_prices = scaler.inverse_transform(dummy)[:, 0]

    dummy2 = np.zeros((len(y), 5))
    dummy2[:, 0] = y
    actual_prices = scaler.inverse_transform(dummy2)[:, 0]

    # Metrics
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    print(f"\n{ticker} Model Evaluation:")
    print(f"RMSE: ${rmse:.2f} average error per prediction")
    print(f"MAPE: {mape:.2f}% average % error")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(actual_prices, label='Actual', color='blue')
    plt.plot(predicted_prices, label='Predicted', color='orange')
    plt.title(f'{ticker} — Actual vs Predicted | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Test on any stock
evaluate_stock("AAPL")