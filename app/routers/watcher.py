import threading
import time
import schedule as schedule_lib
from fastapi import APIRouter
from app.services import watcher as watcher_svc
from app.services.watcher import (
    run_set,
    check_yesterday_predictions,
    get_accuracy_history,
    buy_prices,
    buy_counter,
)

router = APIRouter(prefix="/watcher", tags=["Watcher"])

_running = False
_thread: threading.Thread | None = None


def _schedule_loop():
    while _running:
        schedule_lib.run_pending()
        time.sleep(1)


@router.post("/launch", summary="Launch the full trading bot (run now + schedule every 10 min)")
def launch_bot():
    global _running, _thread

    if _running:
        return {"status": "already running"}

    _running = True

    def start():
        check_yesterday_predictions()
        run_set()
        schedule_lib.every(10).minutes.do(run_set)
        _schedule_loop()

    _thread = threading.Thread(target=start, daemon=True)
    _thread.start()

    return {
        "status":  "launched",
        "message": "Checked yesterday's predictions → ran today's set → scheduled every 10 min",
    }


@router.post("/run", summary="Trigger one trading scan immediately")
def run_once():
    result = run_set()
    return {
        "status":      result if result else "run complete",
        "buy_prices":  buy_prices,
        "buy_counter": buy_counter,
    }


@router.post("/start", summary="Start the background scheduling loop (run_set every 10 min)")
def start_watcher():
    global _running, _thread

    if _running:
        return {"status": "already running"}

    _running = True
    schedule_lib.every(10).minutes.do(run_set)
    _thread = threading.Thread(target=_schedule_loop, daemon=True)
    _thread.start()

    return {"status": "started"}


@router.post("/stop", summary="Stop the background scheduling loop")
def stop_watcher():
    global _running
    _running = False
    schedule_lib.clear()
    return {"status": "stopped"}


@router.get("/status", summary="Show running state and current positions")
def watcher_status():
    return {
        "running":     _running,
        "buy_prices":  buy_prices,
        "buy_counter": buy_counter,
    }


@router.post("/check-accuracy", summary="Verify yesterday's predictions against actual prices")
def check_accuracy():
    correct, total = check_yesterday_predictions()
    accuracy = round((correct / total) * 100, 2) if total > 0 else None
    return {"correct": correct, "total": total, "accuracy_pct": accuracy}


@router.get("/accuracy-history", summary="Show running directional accuracy across all sessions")
def accuracy_history():
    history = get_accuracy_history()
    if not history:
        return {"history": [], "overall_accuracy_pct": None}

    total_correct = sum(h["correct"] for h in history)
    total_total   = sum(h["total"]   for h in history)
    overall       = round((total_correct / total_total) * 100, 2) if total_total > 0 else None

    return {"history": history[-20:], "overall_accuracy_pct": overall}
