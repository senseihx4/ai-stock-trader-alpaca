from fastapi import FastAPI
from predictor import predict_next_day
from trader import buy_stock, sell_stock
from watcher import run_set, buy_counter, buy_prices
import threading
import schedule
import time

app = FastAPI()

watcher_running = False
watcher_thread = None

@app.get("/")
def root():
    return {"message": "Stock Trading Bot is running!"}

@app.post("/trade/{ticker}")
def trade(ticker: str, qty: int = 5):
    signal, current, predicted = predict_next_day(ticker)
    
    if signal == "BUY":
        order = buy_stock(ticker, qty)
    else:
        order = sell_stock(ticker, qty)
    
    return {
        "ticker": ticker,
        "signal": signal,
        "current_price": round(current, 2),
        "predicted_price": round(predicted, 2),
        "qty": qty,
        "status": "order placed"
    }


@app.post("/watcher/run")
def run_watcher_once():
    result = run_set()
    status = result if result else "run complete"
    return {"status": status, "buy_prices": buy_prices, "buy_counter": buy_counter}


@app.post("/watcher/start")
def start_watcher():
    global watcher_running, watcher_thread
    if watcher_running:
        return {"status": "already running"}
    watcher_running = True

    def loop():
        while watcher_running:
            schedule.run_pending()
            time.sleep(1)

    watcher_thread = threading.Thread(target=loop, daemon=True)
    watcher_thread.start()
    return {"status": "started"}


@app.post("/watcher/stop")
def stop_watcher():
    global watcher_running
    watcher_running = False
    return {"status": "stopped"}


@app.get("/watcher/status")
def watcher_status():
    return {
        "running": watcher_running,
        "buy_prices": buy_prices,
        "buy_counter": buy_counter,
    }
