from fastapi import APIRouter
from app.services.watcher import predict_next_day
from app.services.trader import buy_stock, sell_stock, get_account

router = APIRouter(prefix="/trade", tags=["Trade"])


@router.get("/account", summary="Get Alpaca account info")
def account():
    return get_account()


@router.post("/{ticker}", summary="Predict and place a single order for a ticker")
def trade(ticker: str, qty: int = 1):
    signal, current, predicted = predict_next_day(ticker.upper())

    if signal == "BUY":
        order = buy_stock(ticker.upper(), qty)
    else:
        order = sell_stock(ticker.upper(), qty)

    return {
        "ticker":          ticker.upper(),
        "signal":          signal,
        "current_price":   round(current, 2),
        "predicted_price": round(predicted, 2),
        "qty":             qty,
        "status":          "order placed",
    }
