import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from app.routers import trade, watcher

app = FastAPI(
    title="Automatic Stock Trading Bot",
    description=(
        "AI-powered stock trading bot using LSTM price prediction and Alpaca paper trading.\n\n"
        "**Quick start:**\n"
        "1. `POST /watcher/launch` — starts the full bot (checks yesterday's accuracy → runs today's scan → schedules every 10 min)\n"
        "2. `GET /watcher/status` — see live positions\n"
        "3. `POST /trade/{ticker}` — manually trade a single stock\n"
    ),
    version="1.0.0",
)

app.include_router(trade.router)
app.include_router(watcher.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "running",
        "docs":    "/docs",
        "message": "Visit /docs to launch the bot or trade individual stocks",
    }
