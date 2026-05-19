# 📈 Alpaca Stock Monitor

A real-time stock monitoring tool built on the [Alpaca Markets API](https://alpaca.markets). Track live prices, manage watchlists, stream quotes via WebSocket, and monitor your portfolio — all programmatically.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Fetch a Quote](#fetch-a-quote)
  - [Stream Live Prices](#stream-live-prices)
  - [Monitor Positions](#monitor-positions)
  - [Track Orders](#track-orders)
- [API Reference](#api-reference)
- [Rate Limits](#rate-limits)
- [Paper vs Live Trading](#paper-vs-live-trading)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Real-time quotes** — Latest bid, ask, and last trade price for any US equity
- **WebSocket streaming** — Subscribe to live trades, quotes, and minute bars
- **Portfolio monitoring** — Current positions, market value, unrealized P&L
- **Order tracking** — Open, filled, and cancelled order history
- **Historical bars** — OHLCV data at minute, hour, or daily resolution
- **Paper trading support** — Test strategies without risking real capital

---

## Prerequisites

- Node.js ≥ 18 (or Python ≥ 3.9)
- An [Alpaca account](https://app.alpaca.markets/signup) (free)
- API Key ID and Secret Key from the Alpaca dashboard

> **Note:** Market data endpoints are available on the free plan. Portfolio and order endpoints require a funded or paper trading account.

---

## Installation

### Node.js

```bash
npm install @alpacahq/alpaca-trade-api
```

### Python

```bash
pip install alpaca-py
```

---

## Configuration

Store your credentials in a `.env` file — never commit them to version control.

```env
ALPACA_API_KEY=your_api_key_id
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # or https://api.alpaca.markets for live
```

Load them in your app:

```js
// Node.js
import Alpaca from "@alpacahq/alpaca-trade-api";

const alpaca = new Alpaca({
  keyId: process.env.ALPACA_API_KEY,
  secretKey: process.env.ALPACA_SECRET_KEY,
  paper: true, // set to false for live trading
});
```

```python
# Python
from alpaca.data import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

data_client = StockHistoricalDataClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
)
```

---

## Usage

### Fetch a Quote

Get the latest quote for one or more symbols.

```js
// Node.js
const quote = await alpaca.getLatestQuote("AAPL");
console.log(`AAPL ask: $${quote.ap}`);
```

```python
# Python
from alpaca.data.requests import StockLatestQuoteRequest

request = StockLatestQuoteRequest(symbol_or_symbols=["AAPL", "TSLA"])
quotes = data_client.get_stock_latest_quote(request)
print(quotes["AAPL"].ask_price)
```

---

### Stream Live Prices

Use WebSockets to receive real-time trade and quote updates.

```js
// Node.js
const stream = alpaca.data_stream_v2;

stream.onConnect(() => {
  stream.subscribeForTrades(["AAPL", "MSFT"]);
});

stream.onStockTrade((trade) => {
  console.log(`${trade.S} traded at $${trade.p} (${trade.s} shares)`);
});

stream.connect();
```

```python
# Python
from alpaca.data.live import StockDataStream

wss = StockDataStream(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
)

async def on_trade(trade):
    print(f"{trade.symbol} @ ${trade.price}")

wss.subscribe_trades(on_trade, "AAPL", "TSLA")
wss.run()
```

---

### Monitor Positions

View all open positions and their current market value.

```js
// Node.js
const positions = await alpaca.getPositions();

positions.forEach((p) => {
  console.log(`${p.symbol}: ${p.qty} shares | Market value: $${p.market_value} | P&L: $${p.unrealized_pl}`);
});
```

```python
# Python
trading_client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
    paper=True,
)

positions = trading_client.get_all_positions()
for p in positions:
    print(f"{p.symbol}: {p.qty} shares, unrealized P&L: ${p.unrealized_pl}")
```

---

### Track Orders

List recent orders filtered by status.

```js
// Node.js
const orders = await alpaca.getOrders({ status: "open", limit: 10 });

orders.forEach((o) => {
  console.log(`${o.symbol} ${o.side} ${o.qty} @ ${o.type} — ${o.status}`);
});
```

```python
# Python
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=10)
orders = trading_client.get_orders(request)
for o in orders:
    print(f"{o.symbol} {o.side} {o.qty} shares — {o.status}")
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v2/account` | GET | Account details and portfolio value |
| `/v2/positions` | GET | All open positions |
| `/v2/positions/{symbol}` | GET | Position for a specific symbol |
| `/v2/orders` | GET | List orders |
| `/v2/orders` | POST | Place a new order |
| `/v2/stocks/{symbol}/quotes/latest` | GET | Latest quote |
| `/v2/stocks/{symbol}/trades/latest` | GET | Latest trade |
| `/v2/stocks/{symbol}/bars` | GET | Historical OHLCV bars |

Full API docs: [https://docs.alpaca.markets](https://docs.alpaca.markets)

---

## Rate Limits

| Plan | Data requests | Order requests |
|---|---|---|
| Free (Market Data) | 200 req/min | — |
| Paper / Live | 200 req/min | 200 req/min |

WebSocket connections support up to **30 subscriptions** per connection on the free plan. Upgrade to the Unlimited plan for higher limits.

---

## Paper vs Live Trading

| Feature | Paper (`paper-api.alpaca.markets`) | Live (`api.alpaca.markets`) |
|---|---|---|
| Real money | ❌ | ✅ |
| Real market data | ✅ | ✅ |
| Order execution | Simulated | Real |
| Recommended for | Development & testing | Production |

Always develop and test against the paper environment before switching to live.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

> Built with the [Alpaca Markets API](https://alpaca.markets). Not affiliated with or endorsed by Alpaca Securities LLC.
