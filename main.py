from fastapi import FastAPI
from predictor import predict_next_day
from trader import buy_stock, sell_stock

app = FastAPI()

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