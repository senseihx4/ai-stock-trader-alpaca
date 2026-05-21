import numpy as np
import torch
import pandas as pd
import ta
import yfinance as yf
import pickle
import json
import os
import schedule
from datetime import datetime
from pathlib import Path
from app.ml.model import StockLSTM
from app.services.trader import buy_stock, sell_stock, is_market_open
from app.core.stocks import STOCKS

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE_DIR      = Path(__file__).parent.parent.parent
MODELS_DIR    = BASE_DIR / "models"
LOG_FILE      = BASE_DIR / "prediction_log.json"
ACCURACY_FILE = BASE_DIR / "accuracy_history.json"

MAX_BUYS     = 5
MIN_GAIN_PCT = 1.5

# In-memory state (persists for the lifetime of the server process)
buy_counter = {ticker: 0 for ticker in STOCKS}
buy_prices: dict = {}


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, start: str = "2018-01-01"):
    end = datetime.today().strftime("%Y-%m-%d")
    return yf.download(ticker, start=start, end=end, progress=False)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_next_day(ticker: str):
    model = StockLSTM(input_size=5, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(MODELS_DIR / f"{ticker}_model.pth", map_location=device))
    model.to(device)
    model.eval()

    with open(MODELS_DIR / f"{ticker}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = fetch_data(ticker)
    df.columns = df.columns.get_level_values(0)

    data = pd.DataFrame()
    data["Close"]  = df["Close"].squeeze()
    data["Volume"] = df["Volume"].squeeze()
    data["RSI"]    = ta.momentum.RSIIndicator(data["Close"]).rsi()
    data["MACD"]   = ta.trend.MACD(data["Close"]).macd()
    data["MA20"]   = data["Close"].rolling(window=20).mean()
    data = data.dropna()

    current_price = float(data["Close"].values[-1])
    scaled        = scaler.transform(data)
    last_seq      = torch.tensor(scaled[-30:], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(last_seq.unsqueeze(0)).squeeze().cpu().numpy()

    dummy = pd.DataFrame(
        [[float(pred), 0.0, 0.0, 0.0, 0.0]],
        columns=["Close", "Volume", "RSI", "MACD", "MA20"],
    )
    predicted_price = float(scaler.inverse_transform(dummy)[0][0])
    signal          = "BUY" if predicted_price > current_price else "SELL"

    return signal, current_price, predicted_price


# ── Prediction logging ────────────────────────────────────────────────────────

def load_log() -> dict:
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return {}


def save_log(log: dict):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def log_prediction(ticker: str, current_price: float, predicted_price: float, signal: str):
    log = load_log()
    log[ticker] = {
        "date":               datetime.today().strftime("%Y-%m-%d"),
        "current_price":      current_price,
        "predicted_price":    predicted_price,
        "predicted_direction": signal,
        "actual_next_price":  None,
        "correct":            None,
    }
    save_log(log)


# ── Accuracy verification ─────────────────────────────────────────────────────

def check_yesterday_predictions() -> tuple[int, int]:
    log     = load_log()
    correct = 0
    total   = 0
    results = []

    for ticker, entry in log.items():
        if entry.get("actual_next_price") is not None:
            continue
        try:
            df           = fetch_data(ticker)
            df.columns   = df.columns.get_level_values(0)
            actual_price = float(df["Close"].squeeze().iloc[-1])

            predicted_up = entry["predicted_direction"] == "BUY"
            actually_up  = actual_price > entry["current_price"]
            is_correct   = predicted_up == actually_up

            entry["actual_next_price"] = actual_price
            entry["correct"]           = is_correct

            correct += int(is_correct)
            total   += 1
            results.append({
                "ticker":    ticker,
                "predicted": entry["predicted_direction"],
                "actual":    "↑ UP" if actually_up else "↓ DOWN",
                "correct":   is_correct,
            })
        except Exception as e:
            print(f"  ⚠️  Could not verify {ticker}: {e}")

    save_log(log)

    if total > 0:
        accuracy = (correct / total) * 100
        history  = []
        if ACCURACY_FILE.exists():
            with open(ACCURACY_FILE) as f:
                history = json.load(f)
        history.append({
            "date":     datetime.today().strftime("%Y-%m-%d"),
            "correct":  correct,
            "total":    total,
            "accuracy": round(accuracy, 2),
        })
        with open(ACCURACY_FILE, "w") as f:
            json.dump(history, f, indent=2)

    return correct, total


def get_accuracy_history() -> list:
    if not ACCURACY_FILE.exists():
        return []
    with open(ACCURACY_FILE) as f:
        return json.load(f)


# ── Main trading loop ─────────────────────────────────────────────────────────

def run_set() -> str | None:
    if not is_market_open():
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] Market closed — skipping."
        print(msg)
        return msg

    print(f"\n{'═' * 50}")
    print(f"  🤖  RUN SET  │  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 50}")

    for ticker in STOCKS:
        try:
            signal, current, predicted = predict_next_day(ticker)
            gain_pct = ((predicted - current) / current) * 100

            log_prediction(ticker, current, predicted, signal)

            if ticker in buy_prices:
                profit_pct = ((current - buy_prices[ticker]) / buy_prices[ticker]) * 100
                if profit_pct >= MIN_GAIN_PCT:
                    sell_stock(ticker, 1)
                    print(f"  💰 SELL  {ticker:6s} │ profit {profit_pct:.2f}%")
                    del buy_prices[ticker]
                    buy_counter[ticker] = 0
                    continue

            if signal == "BUY" and gain_pct >= MIN_GAIN_PCT and buy_counter[ticker] < MAX_BUYS:
                buy_stock(ticker, 1)
                buy_prices[ticker]   = current
                buy_counter[ticker] += 1
                print(f"  ✅ BUY   {ticker:6s} │ ${current:.2f} → ${predicted:.2f} │ +{gain_pct:.2f}%")
            else:
                arrow = "📈" if signal == "BUY" else "📉"
                sign  = "+" if gain_pct >= 0 else ""
                print(f"  {arrow}  {ticker:6s} │ ${current:.2f} │ {sign}{gain_pct:.2f}% │ no trade")

        except Exception as e:
            print(f"  ❌  {ticker}: {e}")

    print(f"{'═' * 50}\n")
    return None
