import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import pandas as pd
from sklearn.metrics import mean_squared_error
from app.ml.model import StockLSTM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def evaluate_stock(ticker):
    print(f"\n{'═' * 55}")
    print(f"  📊  EVALUATING: {ticker}")
    print(f"{'═' * 55}")

    model = StockLSTM(input_size=5, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(MODELS_DIR / f"{ticker}_model.pth", map_location=device))
    model.to(device)
    model.eval()

    with open(MODELS_DIR / f"{ticker}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = yf.download(ticker, start="2024-01-01", end="2026-01-01", progress=False)
    df.columns = df.columns.get_level_values(0)

    data = pd.DataFrame()
    data['Close']  = df['Close'].squeeze()
    data['Volume'] = df['Volume'].squeeze()
    data['RSI']    = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD']   = ta.trend.MACD(data['Close']).macd()
    data['MA20']   = data['Close'].rolling(window=20).mean()
    data = data.dropna()

    scaled = scaler.transform(data)

    X, y = [], []
    for i in range(len(scaled) - 30):
        X.append(scaled[i:i + 30])
        y.append(scaled[i + 30][0])

    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = np.array(y)

    with torch.no_grad():
        preds = model(X).squeeze().cpu().numpy()

    dummy = np.zeros((len(preds), 5))
    dummy[:, 0] = preds
    predicted_prices = scaler.inverse_transform(dummy)[:, 0]

    dummy2 = np.zeros((len(y), 5))
    dummy2[:, 0] = y
    actual_prices = scaler.inverse_transform(dummy2)[:, 0]

    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    current_prices  = actual_prices[:-1]
    next_actual     = actual_prices[1:]
    next_predicted  = predicted_prices[:-1]

    predicted_up = next_predicted > current_prices
    actually_up  = next_actual    > current_prices

    correct              = np.sum(predicted_up == actually_up)
    total                = len(current_prices)
    directional_accuracy = (correct / total) * 100

    buy_signals  = predicted_up == True
    sell_signals = predicted_up == False
    buy_correct  = np.sum((predicted_up == actually_up) & buy_signals)
    sell_correct = np.sum((predicted_up == actually_up) & sell_signals)
    buy_total    = np.sum(buy_signals)
    sell_total   = np.sum(sell_signals)
    buy_accuracy  = (buy_correct  / buy_total)  * 100 if buy_total  > 0 else 0
    sell_accuracy = (sell_correct / sell_total) * 100 if sell_total > 0 else 0

    gains = []
    for i in range(len(current_prices)):
        if predicted_up[i]:
            gain_pct = ((next_actual[i] - current_prices[i]) / current_prices[i]) * 100
            gains.append(gain_pct)

    avg_gain       = np.mean(gains) if gains else 0
    profitable_pct = (np.sum(np.array(gains) > 0) / len(gains)) * 100 if gains else 0

    print(f"\n  PRICE ACCURACY")
    print(f"  {'RMSE':<25} ${rmse:.2f}")
    print(f"  {'MAPE':<25} {mape:.2f}%")
    print(f"\n  DIRECTIONAL ACCURACY")
    print(f"  {'Overall':<25} {correct}/{total} = {directional_accuracy:.1f}%")
    print(f"  {'BUY signals correct':<25} {buy_correct}/{buy_total} = {buy_accuracy:.1f}%")
    print(f"  {'SELL signals correct':<25} {sell_correct}/{sell_total} = {sell_accuracy:.1f}%")
    print(f"\n  PROFIT SIMULATION")
    print(f"  {'Avg gain per trade':<25} {avg_gain:+.2f}%")
    print(f"  {'% of BUY trades profitable':<25} {profitable_pct:.1f}%")

    if directional_accuracy >= 60:
        verdict = "🟢  Strong"
    elif directional_accuracy >= 55:
        verdict = "🟡  Decent"
    elif directional_accuracy >= 50:
        verdict = "🟠  Weak"
    else:
        verdict = "🔴  Bad — consider retraining"
    print(f"\n  VERDICT: {verdict}")
    print(f"{'═' * 55}\n")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'{ticker} — Model Evaluation', fontsize=14, fontweight='bold')
    axes[0].plot(actual_prices,    label='Actual',    color='steelblue',  linewidth=1.5)
    axes[0].plot(predicted_prices, label='Predicted', color='darkorange', linewidth=1.5, alpha=0.8)
    axes[0].set_title(f'Price: RMSE ${rmse:.2f} | MAPE {mape:.2f}% | Directional {directional_accuracy:.1f}%')
    axes[0].legend()
    axes[0].set_ylabel('Price ($)')
    colors = ['green' if c else 'red' for c in (predicted_up == actually_up)]
    axes[1].bar(range(len(colors)), [1] * len(colors), color=colors, alpha=0.6, width=1.0)
    axes[1].set_title(f'Directional calls — green=correct, red=wrong ({directional_accuracy:.1f}%)')
    axes[1].set_ylabel('Each prediction')
    axes[1].set_xlabel('Trading days (2024–2026)')
    axes[1].set_yticks([])
    plt.tight_layout()
    plt.show()

    return {
        'ticker':                ticker,
        'directional_accuracy':  round(directional_accuracy, 2),
        'buy_accuracy':          round(buy_accuracy, 2),
        'sell_accuracy':         round(sell_accuracy, 2),
        'rmse':                  round(rmse, 2),
        'mape':                  round(mape, 2),
        'avg_gain_per_trade':    round(avg_gain, 2),
        'profitable_trades_pct': round(profitable_pct, 2),
    }


def evaluate_all(tickers):
    results = []
    for ticker in tickers:
        try:
            r = evaluate_stock(ticker)
            results.append(r)
        except Exception as e:
            print(f"❌ {ticker} failed: {e}")

    if not results:
        return

    print(f"\n{'═' * 75}")
    print(f"  SUMMARY — {len(results)} stocks evaluated")
    print(f"{'═' * 75}")
    print(f"  {'Ticker':<8} {'Dir Acc':>8} {'BUY Acc':>8} {'SELL Acc':>9} {'Avg Gain':>9} {'Verdict':>10}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*9} {'─'*10}")

    for r in sorted(results, key=lambda x: x['directional_accuracy'], reverse=True):
        if r['directional_accuracy'] >= 60:
            verdict = "🟢 Strong"
        elif r['directional_accuracy'] >= 55:
            verdict = "🟡 Decent"
        elif r['directional_accuracy'] >= 50:
            verdict = "🟠 Weak"
        else:
            verdict = "🔴 Bad"
        print(
            f"  {r['ticker']:<8} "
            f"{r['directional_accuracy']:>7.1f}% "
            f"{r['buy_accuracy']:>7.1f}% "
            f"{r['sell_accuracy']:>8.1f}% "
            f"{r['avg_gain_per_trade']:>+8.2f}% "
            f"  {verdict}"
        )

    avg_dir = np.mean([r['directional_accuracy'] for r in results])
    print(f"{'─' * 75}")
    print(f"  {'AVERAGE':<8} {avg_dir:>7.1f}%")
    print(f"{'═' * 75}\n")


if __name__ == "__main__":
    evaluate_stock("AAPL")
