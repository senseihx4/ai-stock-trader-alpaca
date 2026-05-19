import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Connect to Alpaca paper trading
client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def get_account():
    account = client.get_account()
    print(f"Status: {account.status}")
    print(f"Cash: ${account.cash}")
    return account

def buy_stock(symbol, qty):
    order = client.submit_order(
        MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
    )
    print(f"BUY order placed: {qty} shares of {symbol}")
    return order

def sell_stock(symbol, qty):
    order = client.submit_order(
        MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
    )
    print(f"SELL order placed: {qty} shares of {symbol}")
    return order


if __name__ == "__main__":
    get_account()

# # Add at bottom of trader.py
# if __name__ == "__main__":
#     client.cancel_orders()
#     print("All orders cancelled!")