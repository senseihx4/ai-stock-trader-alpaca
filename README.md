# Automatic Stock Trading Bot

An AI-powered stock trading bot that uses a custom LSTM neural network to predict next-day price **direction** and automatically places buy/sell orders through the [Alpaca](https://alpaca.markets) paper trading API. Fully controllable via a FastAPI server — start the bot, check accuracy, and monitor live positions from your browser at `/docs`.

---

## What Makes This Project Different

Most stock prediction models report high price accuracy (97%+) and call it done. This project goes further:

- **Identified the misleading metric** — a model predicting "tomorrow ≈ today" scores 97% price accuracy while being useless for trading
- **Built a directional accuracy evaluator** — the metric that actually matters: did the model correctly predict UP or DOWN?
- **Discovered real model performance** — AAPL tested at 52% overall, with BUY signals at 57.8% and SELL signals at 45.5%
- **Acted on the findings** — proposed three concrete improvements: BUY-only signals, better features, and a directional loss function

---

## Screenshots

### AAPL model evaluation — predicted vs actual price + directional accuracy bar chart
![AAPL evaluation](<img width="1456" height="833" alt="image" src="https://github.com/user-attachments/assets/cbdab2ae-357f-4695-bf8f-c269e6aae846" />
)

### Live positions on Alpaca paper trading dashboard
![Alpaca positions](<img width="1454" height="840" alt="image" src="https://github.com/user-attachments/assets/753e1e61-89aa-4440-a982-a2d7ed859300" />
)

### FastAPI Swagger UI — all endpoints
![Swagger UI](<img width="1372" height="892" alt="image" src="https://github.com/user-attachments/assets/98d0d62e-c6da-4624-a2bc-37dc86b84740" />
)

### Live API response from `/watcher/run`
![Swagger response](<img width="1372" height="892" alt="image" src="https://github.com/user-attachments/assets/94fcd8bf-10e7-41d3-8d51-c06f2fe46326" />)


---

## How It Works

1. **Train** — An LSTM model is trained per stock on 8 years of daily OHLCV data with five features: Close price, Volume, RSI, MACD, and a 20-day moving average.
2. **Predict** — Each day the model predicts tomorrow's closing price. If it expects a gain above the 1.5% threshold it signals `BUY`, otherwise `SELL`.
3. **Trade** — The bot places real paper orders via Alpaca and tracks open positions in memory.
4. **Verify** — The next day it fetches actual prices and scores yesterday's predictions for directional accuracy, building a running history saved to `accuracy_history.json`.

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
│   │   ├── model.py         # LSTM model definition (2-layer, hidden_size=64)
│   │   └── evaluate.py      # directional accuracy backtesting tool
│   ├── services/
│   │   ├── trader.py        # Alpaca API wrapper (buy / sell / account)
│   │   └── watcher.py       # prediction engine + trading loop logic
│   └── core/
│       └── stocks.py        # master list of ~100 tracked tickers
├── scripts/
│   ├── train_all.py         # train one LSTM per stock (100 epochs each)
│   ├── download.py          # bulk-download 8y of OHLCV data
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
- An [Alpaca account](https://app.alpaca.markets/signup) (free paper trading works)
- API Key + Secret from the Alpaca dashboard
- Apple Silicon Mac recommended — the bot uses MPS acceleration automatically (falls back to CPU on other hardware)

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

This downloads 8 years of daily data, trains a 2-layer LSTM for each ticker in `app/core/stocks.py`, and saves the weights and scaler to `models/`. Training covers major market events including the 2020 COVID crash, 2021 bull market, and 2022 bear market.

> **Note on the 97% accuracy figure:** The models report ~97% price accuracy during training. This is a known misleading metric — see the Evaluation section below for the real performance numbers.

---

## Starting the Server

```bash
uvicorn app.main:app --reload
```

Then open **[http://localhost:8000/docs](http://localhost:8000/docs)** in your browser.

---

## Quick Start (3 steps in Swagger UI)

Once the server is running, go to `http://127.0.0.1:8000/docs` and do these three steps in order:

**Step 1 — Confirm Alpaca is connected**
`GET /trade/account` → Try it out → Execute
Shows your paper trading cash balance and account status.

**Step 2 — Launch the bot**
`POST /watcher/launch` → Try it out → Execute
Does three things automatically: checks yesterday's prediction accuracy → runs a full scan of all stocks right now → schedules itself to repeat every 10 minutes.

**Step 3 — Confirm it's running**
`GET /watcher/status` → Try it out → Execute
Shows which positions the bot has opened and confirms the scheduler is active.

The bot runs in the background from there. Hit `GET /watcher/status` anytime to check what it's doing.

---

## API Endpoints

### Watcher (Bot Control)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/watcher/launch` | **Main button** — checks yesterday's accuracy → runs today's scan → schedules every 10 min |
| `POST` | `/watcher/run` | Trigger one immediate scan of all stocks right now, without setting up the schedule |
| `POST` | `/watcher/start` | Start the repeating 10-min scheduler only — does NOT run an immediate scan first |
| `POST` | `/watcher/stop` | Stop the background scheduler and clear the schedule |
| `GET`  | `/watcher/status` | Show running state, current positions (buy prices), and buy counts per stock |
| `POST` | `/watcher/check-accuracy` | Compare yesterday's predictions against actual prices — returns correct/total/accuracy% |
| `GET`  | `/watcher/accuracy-history` | Full directional accuracy log across all past sessions |

### Trade (Single Orders)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/trade/account` | Return Alpaca paper trading account info — cash balance and account status |
| `POST` | `/trade/{ticker}` | Run the LSTM prediction for one stock and immediately place a BUY or SELL order |

---

## Trading Logic

| Signal | Condition |
|--------|-----------|
| `BUY` | Model predicts tomorrow's price > 1.5% above today, and position limit not reached (max 5 buys per stock) |
| `SELL` | A held position has gained ≥ 1.5% since purchase |
| No trade | Predicted gain is below the 1.5% threshold |

The bot runs every 10 minutes during US market hours (9:30 AM – 4:00 PM ET) and skips automatically when the market is closed.

---

## Model Evaluation

Run the evaluator on any trained model:

```python
from app.ml.evaluate import evaluate_stock, evaluate_all

# Single stock — shows full metrics + two charts
evaluate_stock("AAPL")

# All stocks — ranked summary table
from app.core.stocks import STOCKS
evaluate_all(STOCKS)
```

### What the evaluator measures

**Price accuracy (misleading)**
- RMSE — average dollar error per prediction
- MAPE — average percentage error
- These look great (~97%) but mean nothing for trading — a model that just copies yesterday's price scores equally well

**Directional accuracy (what actually matters)**
- Overall — what % of UP/DOWN calls were correct
- BUY accuracy — how often "predicted UP" actually went UP
- SELL accuracy — how often "predicted DOWN" actually went DOWN
- Benchmark: 50% = coin flip, 55%+ = has an edge, 60%+ = strong

**Profit simulation**
- Average gain per BUY trade if every signal was followed
- % of BUY trades that were profitable

### AAPL results (2024–2026 test period)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Price accuracy (MAPE) | 2.83% | Looks like 97% accuracy — misleading |
| Directional accuracy | 52.0% | Barely above random |
| BUY signal accuracy | 57.8% | Has a genuine edge |
| SELL signal accuracy | 45.5% | Worse than random — filtered out |
| Avg gain per BUY trade | +0.23% | Positive but thin |

**Key finding:** The high price accuracy is misleading — the model learned to lag the actual price line rather than predict direction. BUY signals have a genuine edge (57.8%) due to the stock's long-term upward trend. SELL signals test below random (45.5%) and are excluded from live trading.

---

## Known Limitations and Planned Improvements

**1. BUY-only signals (implemented)**
SELL signals tested at 45.5% — below random. Filtering them out and only trading BUY signals immediately improves overall signal quality. The bot only enters new positions on BUY signals; exits are based on the 1.5% profit target, not model direction.

**2. Add stronger features**
Current features (Close, Volume, RSI, MACD, MA20) are all backward-looking. Adding rate-of-change, Bollinger Bands, and volume trend indicators would give the model more predictive signal.

**3. Change the loss function**
The model was trained to minimise price error (MSE loss). Retraining with a directional loss — penalising wrong UP/DOWN calls rather than dollar distance — would directly optimise for the metric that matters for trading.

---

## Paper vs Live Trading

| Feature | Paper | Live |
|---------|-------|------|
| Real money | No | Yes |
| Real market data | Yes | Yes |
| Order execution | Simulated | Real |
| URL | `paper-api.alpaca.markets` | `api.alpaca.markets` |

Validate directional accuracy over several weeks of paper trading before considering live deployment.

---

## Emergency Reset

Cancel all open orders and close all positions immediately:

```bash
python scripts/reset.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML model | PyTorch — 2-layer LSTM, input_size=5, hidden_size=64 |
| Features | ta-lib — RSI, MACD, MA20 |
| Data | yfinance — 8 years of daily OHLCV |
| Preprocessing | scikit-learn MinMaxScaler |
| Broker API | Alpaca Markets (paper trading) |
| Server | FastAPI + uvicorn |
| Scheduling | schedule — runs every 10 min during market hours |
| Acceleration | Apple MPS (Metal Performance Shaders) on Apple Silicon |

---

## License

MIT License.

---

> Built with PyTorch, FastAPI, and the Alpaca Markets API. Not financial advice. Not affiliated with or endorsed by Alpaca Securities LLC.
