import numpy as np
import torch
import pandas as pd
import ta
import yfinance as yf
import pickle
import json
import os
import time
import schedule
from datetime import datetime
from model import StockLSTM
from trader import buy_stock, sell_stock, is_market_open
from stocks import STOCKS

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────────────
MAX_BUYS        = 5
MIN_GAIN_PCT    = 1.5   # minimum predicted gain % to trigger a BUY
LOG_FILE        = "prediction_log.json"
ACCURACY_FILE   = "accuracy_history.json"

# ── State ─────────────────────────────────────────────────────────────────────
buy_counter = {ticker: 0 for ticker in STOCKS}
buy_prices  = {}

# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_data(ticker, start="2018-01-01"):
    end = datetime.today().strftime("%Y-%m-%d")
    df  = yf.download(ticker, start=start, end=end, progress=False)
    return df

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_next_day(ticker):
    model = StockLSTM(input_size=5, hidden_size=64, num_layers=2)
    model.load_state_dict(
        torch.load(f"models/{ticker}_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    with open(f"models/{ticker}_scaler.pkl", "rb") as f:
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
        columns=["Close", "Volume", "RSI", "MACD", "MA20"]
    )
    predicted_price = float(scaler.inverse_transform(dummy)[0][0])
    signal          = "BUY" if predicted_price > current_price else "SELL"

    return signal, current_price, predicted_price

# ── Directional accuracy logging ──────────────────────────────────────────────
def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def log_prediction(ticker, current_price, predicted_price, signal):
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

def check_yesterday_predictions():
    """
    Fetch today's actual price for every logged prediction that hasn't been
    verified yet, then compute and print directional accuracy.
    """
    log     = load_log()
    correct = 0
    total   = 0
    results = []

    for ticker, entry in log.items():
        if entry.get("actual_next_price") is not None:
            continue  # already verified

        try:
            df           = fetch_data(ticker)
            df.columns   = df.columns.get_level_values(0)
            actual_price = float(df["Close"].squeeze().iloc[-1])

            predicted_up  = entry["predicted_direction"] == "BUY"
            actually_up   = actual_price > entry["current_price"]
            is_correct    = predicted_up == actually_up

            entry["actual_next_price"] = actual_price
            entry["correct"]           = is_correct

            correct += int(is_correct)
            total   += 1

            results.append({
                "ticker":    ticker,
                "predicted": entry["predicted_direction"],
                "actual":    "↑ UP"   if actually_up else "↓ DOWN",
                "correct":   "✅"     if is_correct  else "❌",
            })
        except Exception as e:
            print(f"  ⚠️  Could not verify {ticker}: {e}")

    save_log(log)

    print("\n" + "─" * 50)
    print("📊  YESTERDAY'S DIRECTIONAL ACCURACY CHECK")
    print("─" * 50)

    if total == 0:
        print("  No new predictions to verify yet.")
    else:
        for r in results:
            print(f"  {r['correct']}  {r['ticker']:6s} │ predicted {r['predicted']:4s} │ actual {r['actual']}")

        accuracy = (correct / total) * 100
        print("─" * 50)
        print(f"  Result : {correct}/{total} correct  →  {accuracy:.1f}% directional accuracy")

        # Save running history
        history = []
        if os.path.exists(ACCURACY_FILE):
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

        # Benchmark message
        if accuracy >= 60:
            print("  🟢  Strong signal — model is beating the market")
        elif accuracy >= 55:
            print("  🟡  Decent — model has a slight edge")
        else:
            print("  🔴  Weak — close to random (50%). Consider retraining")

    print("─" * 50 + "\n")
    return correct, total

def print_accuracy_history():
    """Print running accuracy across all sessions."""
    if not os.path.exists(ACCURACY_FILE):
        print("  No accuracy history yet.\n")
        return

    with open(ACCURACY_FILE) as f:
        history = json.load(f)

    total_correct = sum(h["correct"] for h in history)
    total_total   = sum(h["total"]   for h in history)

    print("\n" + "─" * 50)
    print("📈  OVERALL ACCURACY HISTORY")
    print("─" * 50)
    for h in history[-10:]:  # show last 10 sessions
        bar = "█" * int(h["accuracy"] / 5)
        print(f"  {h['date']}  {h['accuracy']:5.1f}%  {bar}")

    if total_total > 0:
        overall = (total_correct / total_total) * 100
        print("─" * 50)
        print(f"  Overall: {total_correct}/{total_total} = {overall:.1f}%")
    print("─" * 50 + "\n")

# ── Main trading loop ─────────────────────────────────────────────────────────
def run_set():
    if not is_market_open():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed — skipping.")
        return

    print(f"\n{'═' * 50}")
    print(f"  🤖  RUN SET  │  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 50}")

    for ticker in STOCKS:
        try:
            signal, current, predicted = predict_next_day(ticker)
            gain_pct = ((predicted - current) / current) * 100

            # ── Log prediction for accuracy tracking ──
            log_prediction(ticker, current, predicted, signal)

            # ── Check for profit sell ─────────────────
            if ticker in buy_prices:
                profit_pct = ((current - buy_prices[ticker]) / buy_prices[ticker]) * 100
                if profit_pct >= MIN_GAIN_PCT:
                    sell_stock(ticker, 1)
                    print(
                        f"  💰 SELL  {ticker:6s} │ bought ${buy_prices[ticker]:.2f} "
                        f"│ now ${current:.2f} │ profit {profit_pct:.2f}%"
                    )
                    del buy_prices[ticker]
                    buy_counter[ticker] = 0
                    continue

            # ── Buy signal ────────────────────────────
            if (
                signal == "BUY"
                and gain_pct >= MIN_GAIN_PCT
                and buy_counter[ticker] < MAX_BUYS
            ):
                buy_stock(ticker, 1)
                buy_prices[ticker]       = current
                buy_counter[ticker]     += 1
                print(
                    f"  ✅ BUY   {ticker:6s} │ ${current:.2f} → ${predicted:.2f} "
                    f"│ +{gain_pct:.2f}%  (buy #{buy_counter[ticker]})"
                )

            # ── No trade ──────────────────────────────
            else:
                arrow = "📈" if signal == "BUY" else "📉"
                sign  = "+" if gain_pct >= 0 else ""
                print(
                    f"  {arrow}  {ticker:6s} │ ${current:.2f} "
                    f"│ {sign}{gain_pct:.2f}% │ no trade"
                )

        except Exception as e:
            print(f"  ❌  {ticker}: {e}")

    print(f"{'═' * 50}\n")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖  Trading bot started!")
    print(f"    Device  : {device}")
    print(f"    Stocks  : {len(STOCKS)}")
    print(f"    Max buys: {MAX_BUYS} per stock")
    print(f"    Min gain: {MIN_GAIN_PCT}%\n")

    # Step 1 — Check yesterday's predictions
    check_yesterday_predictions()

    # Step 2 — Print running accuracy history
    print_accuracy_history()

    # Step 3 — Run today's first set immediately
    run_set()

    # Step 4 — Schedule every 10 minutes
    schedule.every(10).minutes.do(run_set)

    while True:
        schedule.run_pending()
        time.sleep(1)