import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta


def prepare_data(df, seq_length=30):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = df[['Close', 'Volume']].copy()
    data['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    data['MACD'] = ta.trend.MACD(df['Close']).macd()
    data['MA20'] = df['Close'].rolling(window=20).mean()
    data = data.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i + seq_length])
        y.append(scaled[i + seq_length][0])

    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(device)
    return X, y, scaler


class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


StockLSTM = LSTMModel
