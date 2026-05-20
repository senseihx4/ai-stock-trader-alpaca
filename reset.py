from alpaca.trading.client import TradingClient
from alpaca.trading.requests import ClosePositionRequest
import os
from dotenv import load_dotenv
import time

load_dotenv()
client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

# Cancel all orders first
print("Cancelling all orders...")
client.cancel_orders()
time.sleep(2)

# Get all positions and close one by one
print("Closing all positions...")
positions = client.get_all_positions()
for position in positions:
    try:
        client.close_position(position.symbol)
        print(f"✅ Closed {position.symbol}")
        time.sleep(0.3)  # small delay to avoid rate limit
    except Exception as e:
        print(f"❌ {position.symbol}: {e}")

print("🎉 Done!")