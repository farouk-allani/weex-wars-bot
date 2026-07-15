"""Macro context — the world outside the crypto chart.

Crypto is a risk asset. It trades off the dollar, real yields, equity risk appetite
and the yen carry, and in the last month it has been driven almost entirely by Fed
repricing. A bot that sees only RSI cannot know *why* price is moving, so it fades
a macro-driven rally and gets run over. That is precisely what ours did.

We take the macro STATE rather than news about it. "Cooler CPI -> rate-cut bets ->
BTC up" registers as: short yields down, dollar down, equities up — and those move
hours before a journalist writes the headline. The state is also free, hourly, and
fully historical, which means the replay harness can actually test whether it helps.

Deliberately reports numbers, not opinions. The last time this file's author told a
model what a number "meant", it went 92% short and lost 18%.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

CACHE_DIR = Path("data/cache/macro")

# Traditional markets close nights/weekends while crypto does not, so every read is
# "last known value at or before now" — a stale-but-real level, never a peek forward.
TICKERS = {
    "dxy": ("DX-Y.NYB", "US dollar index"),
    "us_short_yield": ("^IRX", "US 13-week T-bill yield (Fed expectations proxy)"),
    "us_10y_yield": ("^TNX", "US 10-year Treasury yield"),
    "sp500": ("^GSPC", "S&P 500"),
    "nasdaq": ("^IXIC", "Nasdaq Composite"),
    "vix": ("^VIX", "equity volatility / fear index"),
    "gold": ("GC=F", "gold futures"),
    "oil": ("CL=F", "WTI crude futures"),
    "nikkei": ("^N225", "Nikkei 225 (Japan)"),
    "usdjpy": ("JPY=X", "USD/JPY (yen carry)"),
}


def fetch_macro_history(days: int = 60) -> dict[str, dict[int, float]]:
    """Hourly closes per series, keyed by epoch-ms. Cached — yfinance is slow."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"macro_{days}d.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 1800:
        try:
            raw = json.loads(cache.read_text())
            return {k: {int(t): v for t, v in s.items()} for k, s in raw.items()}
        except Exception:
            pass

    import yfinance as yf

    out: dict[str, dict[int, float]] = {}
    for name, (ticker, _) in TICKERS.items():
        try:
            h = yf.Ticker(ticker).history(period=f"{days}d", interval="1h")
            series = {}
            for ts, row in h.iterrows():
                v = float(row["Close"])
                if np.isfinite(v):
                    series[int(ts.timestamp() * 1000)] = round(v, 4)
            out[name] = series
        except Exception:
            out[name] = {}

    cache.write_text(json.dumps({k: {str(t): v for t, v in s.items()} for k, s in out.items()}))
    return out


def _at(series: dict[int, float], ts_ms: int, max_stale_ms: int = 5 * 86400 * 1000):
    """Last value AT OR BEFORE ts_ms. Never looks ahead.

    Tolerates multi-day staleness on purpose: over a weekend the last real print of
    the S&P *is* the current state of the world, and pretending otherwise would drop
    macro context every Saturday.
    """
    best_t = None
    for t in series:
        if t <= ts_ms and (best_t is None or t > best_t):
            best_t = t
    if best_t is None or ts_ms - best_t > max_stale_ms:
        return None
    return series[best_t]


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return round((a / b - 1) * 100, 2)


def macro_snapshot(hist: dict[str, dict[int, float]], at_ms: int) -> dict:
    """Point-in-time macro state: levels plus 24h / 7d changes."""
    snap: dict = {}
    for name, (_, desc) in TICKERS.items():
        s = hist.get(name) or {}
        now = _at(s, at_ms)
        if now is None:
            continue
        d1 = _at(s, at_ms - 24 * 3600 * 1000)
        d7 = _at(s, at_ms - 7 * 86400 * 1000)
        snap[name] = {
            "level": now,
            "change_24h_pct": _pct(now, d1),
            "change_7d_pct": _pct(now, d7),
            "what": desc,
        }

    snap["_note"] = (
        "Traditional markets close overnight and at weekends; values are the last "
        "print at or before now. Crypto is a risk asset — how it relates to the "
        "dollar, yields, equity risk appetite and the yen is for you to judge."
    )
    return snap


def live_macro() -> dict:
    hist = fetch_macro_history(30)
    return macro_snapshot(hist, int(time.time() * 1000))
