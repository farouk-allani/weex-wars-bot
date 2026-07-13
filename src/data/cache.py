"""Disk cache for OHLCV / funding so backtests don't re-hit the network every run."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from ..core.models import Candle

CACHE_DIR = Path("data/cache")


def _key(symbol: str, timeframe: str, days: int) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}_{timeframe}_{days}d.json"


def save_candles(symbol: str, timeframe: str, days: int, candles: list[Candle]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": time.time(),
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "rows": [
            {
                "ts": c.timestamp.isoformat(),
                "o": c.open,
                "h": c.high,
                "l": c.low,
                "c": c.close,
                "v": c.volume,
            }
            for c in candles
        ],
    }
    path = _key(symbol, timeframe, days)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_candles(
    symbol: str,
    timeframe: str,
    days: int,
    max_age_hours: float = 6.0,
) -> list[Candle] | None:
    path = _key(symbol, timeframe, days)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        age_h = (time.time() - float(payload.get("saved_at", 0))) / 3600
        if age_h > max_age_hours:
            return None
        out = []
        for r in payload.get("rows") or []:
            ts = datetime.fromisoformat(r["ts"])
            out.append(
                Candle(
                    timestamp=ts,
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=float(r["c"]),
                    volume=float(r["v"]),
                )
            )
        return out if len(out) > 50 else None
    except Exception:
        return None


def save_funding(symbol: str, days: int, rates: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"funding_{symbol.replace('/', '_').replace(':', '_')}_{days}d.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"saved_at": time.time(), "rates": {str(k): v for k, v in rates.items()}}, f)


def load_funding(symbol: str, days: int, max_age_hours: float = 6.0) -> dict | None:
    path = CACHE_DIR / f"funding_{symbol.replace('/', '_').replace(':', '_')}_{days}d.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        age_h = (time.time() - float(payload.get("saved_at", 0))) / 3600
        if age_h > max_age_hours:
            return None
        return {int(k): float(v) for k, v in (payload.get("rates") or {}).items()}
    except Exception:
        return None
