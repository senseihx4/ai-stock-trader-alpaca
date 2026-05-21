import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = TradingClient(API_KEY, SECRET_KEY, paper=True)


def is_market_open() -> bool:
    return client.get_clock().is_open


def get_account():
    account = client.get_account()
    return {"status": account.status, "cash": float(account.cash)}


def buy_stock(symbol: str, qty: int):
    order = client.submit_order(
        MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    )
    print(f"BUY order placed: {qty} shares of {symbol}")
    return order


def sell_stock(symbol: str, qty: int):
    order = client.submit_order(
        MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
    )
    print(f"SELL order placed: {qty} shares of {symbol}")
    return order
