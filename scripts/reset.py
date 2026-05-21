"""
Emergency cleanup — cancels all open orders and closes all positions.
Run from project root:
    python scripts/reset.py
"""
import os
import time
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()
client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

print("Cancelling all orders...")
client.cancel_orders()
time.sleep(2)

print("Closing all positions...")
for position in client.get_all_positions():
    try:
        client.close_position(position.symbol)
        print(f"✅ Closed {position.symbol}")
        time.sleep(0.3)
    except Exception as e:
        print(f"❌ {position.symbol}: {e}")

print("🎉 Done!")
