"""Market data fetcher with disk cache (Binance public history for backtests)."""

from __future__ import annotations

from datetime import datetime, timedelta

import ccxt

from ..core.models import Candle
from .cache import load_candles, save_candles, load_funding, save_funding


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    days: int = 90,
    use_cache: bool = True,
) -> list[Candle]:
    if use_cache:
        cached = load_candles(symbol, timeframe, days)
        if cached:
            return cached

    exchange = ccxt.binance({"enableRateLimit": True})
    all_rows = []
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break

    candles = [
        Candle(
            timestamp=datetime.utcfromtimestamp(r[0] / 1000),
            open=r[1],
            high=r[2],
            low=r[3],
            close=r[4],
            volume=r[5],
        )
        for r in all_rows
    ]
    if use_cache and candles:
        save_candles(symbol, timeframe, days, candles)
    return candles


def fetch_funding_map(symbol: str, days: int = 90, use_cache: bool = True) -> dict:
    if use_cache:
        cached = load_funding(symbol, days)
        if cached is not None:
            return cached

    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        data = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        rates = {
            fr["timestamp"]: fr.get("fundingRate", 0.0)
            for fr in (data or [])
            if fr.get("timestamp")
        }
    except Exception:
        rates = {}
    if use_cache:
        save_funding(symbol, days, rates)
    return rates


def interpolate_funding(timestamps, rates: dict) -> list[float]:
    if not rates:
        return [0.0] * len(timestamps)
    sorted_ts = sorted(rates.keys())
    out, last = [], 0.0
    for ts in timestamps:
        ms = int(ts.timestamp() * 1000)
        for t in reversed(sorted_ts):
            if t <= ms:
                last = rates[t]
                break
        out.append(last)
    return out
