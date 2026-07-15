"""Market intelligence the price chart cannot tell you.

Everything here answers one question: *where is the crowd, and is it trapped?*
That is information, not interpretation — which is the thing our TA-only context was
missing. RSI tells you price went up. Positioning tells you who is on the wrong side
of it.

Sources (all free, all with history so the replay harness can validate them):
  Binance futures  openInterestHist              — is a rally new money or short covering?
                   globalLongShortAccountRatio   — where RETAIL is positioned
                   topLongShortPositionRatio     — where TOP TRADERS are positioned
                   takerlongshortRatio           — aggressive order flow
  alternative.me   fear & greed index            — market-wide sentiment

Hard constraint: Binance caps this data at ~30 days. Anything older 400s. That bounds
our replay window, and we say so rather than quietly backtesting on nothing.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

FAPI = "https://fapi.binance.com"
FNG = "https://api.alternative.me/fng/"
CACHE_DIR = Path("data/cache/intel")
MAX_HISTORY_DAYS = 28  # Binance hard limit is 30; stay inside it.


def to_binance(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTCUSDT'"""
    return symbol.split("/")[0] + "USDT"


def _cached(name: str, ttl: int, fetch):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{name}.json"
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    data = fetch()
    if data:
        p.write_text(json.dumps(data))
    return data


def _paged(endpoint: str, symbol: str, days: int, period: str = "1h") -> list[dict]:
    """Walk the window in 500-row pages (the API cap)."""
    days = min(days, MAX_HISTORY_DAYS)
    now = int(time.time() * 1000)
    start = now - days * 86400 * 1000
    step = 500 * 3600 * 1000
    rows: list[dict] = []
    t = start
    while t < now:
        try:
            r = requests.get(
                f"{FAPI}/futures/data/{endpoint}",
                params={"symbol": symbol, "period": period, "limit": 500,
                        "startTime": t, "endTime": min(t + step, now)},
                timeout=20,
            )
            d = r.json()
            if isinstance(d, list):
                rows.extend(d)
            else:
                break
        except Exception:
            break
        t += step
        time.sleep(0.12)  # be polite; these are unauthenticated endpoints
    return rows


def fetch_positioning_history(symbol: str, days: int = MAX_HISTORY_DAYS) -> dict[str, dict]:
    """Positioning series for one symbol, one dict per metric.

    Kept as separate series rather than merged on timestamp: Binance stamps the taker
    endpoint on a different hour boundary than the others, so an exact-timestamp merge
    silently drops it. Each metric is matched to the decision time independently.
    """
    bsym = to_binance(symbol)
    days = min(days, MAX_HISTORY_DAYS)

    def build() -> dict:
        def series(rows, fn) -> dict[str, float]:
            out = {}
            for r in rows:
                ts = int(r.get("timestamp") or 0)
                if ts:
                    try:
                        out[str(ts)] = fn(r)
                    except (TypeError, ValueError):
                        pass
            return out

        return {
            "oi": series(_paged("openInterestHist", bsym, days),
                         lambda r: float(r["sumOpenInterest"])),
            "retail_long": series(_paged("globalLongShortAccountRatio", bsym, days),
                                  lambda r: float(r["longAccount"])),
            "top_long": series(_paged("topLongShortPositionRatio", bsym, days),
                               lambda r: float(r["longAccount"])),
            "taker_buy_sell": series(_paged("takerlongshortRatio", bsym, days),
                                     lambda r: float(r["buySellRatio"])),
        }

    raw = _cached(f"pos_{bsym}_{days}d_v2", ttl=1800, fetch=build) or {}
    return {k: {int(t): v for t, v in s.items()} for k, s in raw.items()}


def fetch_fear_greed(days: int = 90) -> dict[str, int]:
    """Fear & Greed by YYYY-MM-DD. Free, full history, no key."""
    def build():
        r = requests.get(FNG, params={"limit": days, "format": "json"}, timeout=20)
        out = {}
        for row in (r.json() or {}).get("data", []):
            ts = int(row.get("timestamp") or 0)
            if ts:
                day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
                out[day] = int(row.get("value") or 0)
        return out

    return _cached(f"fng_{days}d", ttl=3600, fetch=build) or {}


def _at(series: dict[int, float], ts_ms: int, max_skew_ms: int = 3 * 3600 * 1000):
    """Most recent value AT OR BEFORE ts_ms. Never looks ahead — this is what keeps
    the replay honest."""
    best_t = None
    for t in series:
        if t <= ts_ms and (best_t is None or t > best_t):
            best_t = t
    if best_t is None or ts_ms - best_t > max_skew_ms:
        return None
    return series[best_t]


def _percentile(series: dict[int, float], value: Optional[float], at_ms: int,
                window_ms: int = 30 * 86400 * 1000) -> Optional[int]:
    """Where does `value` sit within its own recent history? (0-100, backward-looking)

    Absolute levels are close to meaningless here. Retail is structurally ~60-70%
    long in crypto perps essentially always, so "retail is 70% long" is a baseline,
    not a contrarian extreme — reading it as one produces a permanent short bias.
    What carries information is whether a reading is unusual *for this instrument*.
    """
    if value is None or not series:
        return None
    hist = [v for t, v in series.items() if at_ms - window_ms <= t <= at_ms]
    if len(hist) < 20:
        return None
    below = sum(1 for v in hist if v < value)
    return int(round(below / len(hist) * 100))


def positioning_snapshot(
    series: dict[str, dict],
    at_ms: int,
    price_change_24h_pct: Optional[float] = None,
    fear_greed: Optional[int] = None,
) -> Optional[dict]:
    """Point-in-time positioning. Reports levels AND their percentile-vs-own-history.

    Deliberately free of editorial reads. An earlier version shipped strings like
    "contrarian bearish" alongside the numbers; the model deferred to the label
    instead of the data and went 92% short. Report; do not interpret.
    """
    if not series:
        return None

    oi = _at(series.get("oi") or {}, at_ms)
    oi_prev = _at(series.get("oi") or {}, at_ms - 24 * 3600 * 1000)
    retail = _at(series.get("retail_long") or {}, at_ms)
    top = _at(series.get("top_long") or {}, at_ms)
    taker = _at(series.get("taker_buy_sell") or {}, at_ms)

    if oi is None and retail is None and top is None:
        return None

    oi_chg = ((oi / oi_prev - 1) * 100) if (oi and oi_prev and oi_prev > 0) else None

    retail_pct = round(retail * 100, 1) if retail is not None else None
    top_pct = round(top * 100, 1) if top is not None else None

    snap: dict = {
        "open_interest": round(oi, 2) if oi else None,
        "oi_change_24h_pct": round(oi_chg, 2) if oi_chg is not None else None,
        "oi_percentile_30d": _percentile(series.get("oi") or {}, oi, at_ms),
        "retail_long_pct": retail_pct,
        # The percentile is the load-bearing number, not the level.
        "retail_long_percentile_30d": _percentile(
            series.get("retail_long") or {}, retail, at_ms),
        "top_trader_long_pct": top_pct,
        "top_trader_long_percentile_30d": _percentile(
            series.get("top_long") or {}, top, at_ms),
        "taker_buy_sell_ratio": round(taker, 3) if taker is not None else None,
        "taker_percentile_30d": _percentile(
            series.get("taker_buy_sell") or {}, taker, at_ms),
    }
    if retail is not None and top is not None:
        snap["retail_minus_top_long_pct"] = round((retail - top) * 100, 1)

    if price_change_24h_pct is not None:
        snap["price_change_24h_pct"] = round(price_change_24h_pct, 2)

    if fear_greed is not None:
        snap["fear_greed"] = fear_greed
    return snap


def live_positioning(symbol: str, price_change_24h_pct: Optional[float] = None) -> Optional[dict]:
    """Current positioning for the live bot.

    Pulls the full 28d window, not just the last few days: the percentiles are the
    whole point, and a percentile computed over 3 days of history is a different
    (and much noisier) statistic than the one the model is told it is reading.
    """
    series = fetch_positioning_history(symbol, days=MAX_HISTORY_DAYS)
    if not series:
        return None
    fng = fetch_fear_greed(2)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return positioning_snapshot(
        series, int(time.time() * 1000), price_change_24h_pct,
        fear_greed=fng.get(today) or (list(fng.values())[0] if fng else None),
    )


def latest_fear_greed() -> Optional[int]:
    fng = fetch_fear_greed(2)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return fng.get(today) or (list(fng.values())[0] if fng else None)
