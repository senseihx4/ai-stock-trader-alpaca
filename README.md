# Automatic Stock Trading Bot

An AI-powered stock trading bot that uses a custom LSTM neural network to predict next-day price direction and automatically places buy/sell orders through the [Alpaca](https://alpaca.markets) paper trading API. Fully controllable via a FastAPI server — start the bot, check accuracy, and monitor live positions from your browser at `/docs`.

---

## How It Works

1. **Train** — An LSTM model is trained per stock on 8 years of daily OHLCV data with five features: Close price, Volume, RSI, MACD, and a 20-day moving average.
2. **Predict** — Each day the model predicts tomorrow's closing price. If it expects a gain above the threshold it signals `BUY`, otherwise `SELL`.
3. **Trade** — The bot places real paper orders via Alpaca and tracks open positions in memory.
4. **Verify** — The next day it fetches actual prices and scores yesterday's predictions (directional accuracy), building a running history.

---

## Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── routers/
│   │   ├── trade.py         # /trade endpoints — single-stock orders
│   │   └── watcher.py       # /watcher endpoints — bot control + accuracy
│   ├── ml/
│   │   ├── model.py         # LSTM model definition
│   │   └── evaluate.py      # offline backtesting tool
│   ├── services/
│   │   ├── trader.py        # Alpaca API wrapper (buy / sell / account)
│   │   └── watcher.py       # prediction engine + trading loop logic
│   └── core/
│       └── stocks.py        # master list of ~60 tracked tickers
├── scripts/
│   ├── train_all.py         # train one LSTM per stock
│   ├── download.py          # bulk-download 10y of OHLCV data
│   └── reset.py             # emergency: cancel orders + close all positions
├── models/                  # saved model weights (.pth) and scalers (.pkl)
├── prediction_log.json      # today's predictions (auto-updated at runtime)
├── accuracy_history.json    # running directional accuracy log
├── .env                     # Alpaca credentials (never commit)
└── requirements.txt
```

---

## Prerequisites

- Python ≥ 3.11
- An [Alpaca account](https://app.alpaca.markets/signup) (free paper trading account works)
- API Key + Secret from the Alpaca dashboard

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_api_key_id
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```

> Always use the paper trading URL while testing. Switch to `https://api.alpaca.markets/v2` only for live trading with real money.

---

## Training the Models

Before running the bot you need a trained model for each stock.

```bash
python -m scripts.train_all
```

This downloads ~8 years of daily data, trains a 2-layer LSTM for each ticker in `app/core/stocks.py` (100 epochs each), and saves the weights + scaler to `models/`.

---

## Starting the Server

```bash
uvicorn app.main:app --reload
```

Then open **[http://localhost:8000/docs](http://localhost:8000/docs)** in your browser.

---

## API Endpoints

### Watcher (Bot Control)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/watcher/launch` | **Start the full bot** — verifies yesterday's predictions, runs today's scan, then schedules every 10 min |
| `POST` | `/watcher/run` | Trigger a single scan immediately |
| `POST` | `/watcher/start` | Start the 10-minute scheduling loop only |
| `POST` | `/watcher/stop` | Stop the scheduling loop |
| `GET`  | `/watcher/status` | Show running state, open positions, and buy counters |
| `POST` | `/watcher/check-accuracy` | Score yesterday's predictions against actual prices |
| `GET`  | `/watcher/accuracy-history` | Full directional accuracy log across all sessions |

### Trade (Single Orders)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/trade/{ticker}` | Predict and immediately place one order for a ticker |
| `GET`  | `/trade/account` | Show Alpaca account status and available cash |

---

## New Features (vs original)

### `/watcher/launch` — One-click bot startup
Previously you had to run `python watcher.py` from the terminal. Now you can launch the entire bot from the Swagger UI:

1. Go to `http://localhost:8000/docs`
2. Click **POST /watcher/launch** → **Try it out** → **Execute**
3. The bot will immediately check yesterday's prediction accuracy, run a full scan, and schedule itself to run every 10 minutes in the background.

### `/watcher/check-accuracy` — Verify predictions
After the market closes, call this endpoint to fetch actual prices and score every prediction made that day. The result is saved to `accuracy_history.json`.

### `/watcher/accuracy-history` — Running accuracy log
Returns the last 20 sessions of directional accuracy (what % of UP/DOWN predictions were correct). Overall lifetime accuracy is included in the response.

### `/trade/account` — Account info via API
Check your Alpaca paper trading balance and account status without leaving the Swagger UI.

---

## Evaluating a Model (Offline)

To backtest a trained model against historical data and plot predicted vs actual prices:

```python
from app.ml.evaluate import evaluate_stock, evaluate_all

# Single stock
evaluate_stock("AAPL")

# Multiple stocks
from app.core.stocks import STOCKS
evaluate_all(STOCKS[:10])
```

Outputs directional accuracy, BUY/SELL accuracy breakdown, simulated profit per trade, and a chart.

---

## Emergency Reset

If you need to cancel all open orders and close all positions immediately:

```bash
python scripts/reset.py
```

---

## Trading Logic

| Signal | Condition |
|--------|-----------|
| `BUY`  | Model predicts tomorrow's price is > 1.5% above today's price, and the position limit for that stock hasn't been reached (max 5 buys per stock) |
| `SELL` | A held position has gained ≥ 1.5% since purchase, OR the model predicts a decline |
| No trade | Predicted gain is below the 1.5% threshold |

The bot runs every 10 minutes during market hours and skips automatically when the market is closed.

---

## Paper vs Live Trading

| Feature | Paper | Live |
|---------|-------|------|
| Real money | ❌ | ✅ |
| Real market data | ✅ | ✅ |
| Order execution | Simulated | Real |
| URL | `paper-api.alpaca.markets` | `api.alpaca.markets` |

Always validate the bot's directional accuracy over several weeks of paper trading before considering live deployment.

---

## License

MIT License.

---

> Built with PyTorch, FastAPI, and the Alpaca Markets API. Not affiliated with or endorsed by Alpaca Securities LLC.
