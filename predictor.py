import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from model import StockLSTM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def fetch_data(ticker, start="2018-01-01", end="2026-01-01"):
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
    # Load model
    model = StockLSTM(input_size=1, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load('stock_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    
    df = fetch_data(ticker)
    last_sequence, scaler, current_price = prepare_data(df)
    
    
    with torch.no_grad():
        pred = model(last_sequence.unsqueeze(0)).squeeze().cpu().numpy()
        predicted_price = scaler.inverse_transform([[pred]])[0][0]
    
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted price: ${predicted_price:.2f}")
    

    if predicted_price > current_price:
        return "BUY", current_price, predicted_price
    else:
        return "SELL", current_price, predicted_price

# Test
if __name__ == "__main__":
    signal, current, predicted = predict_next_day("AAPL")
    print(f"Signal: {signal}")