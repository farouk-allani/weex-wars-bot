"""Liquidation-cascade bounce scan — the forced-flow hypothesis.

Mechanism (not prediction): when leveraged positions hit liquidation, the
exchange market-closes them BY CONTRACT. That flow is forced, not informed,
so the move it causes tends to overshoot and revert once the queue clears.
Same edge family as funding carry (mechanical flows) — the only family that
has survived our testing. See memory: bot-empirical-findings.

Proxy (no free historical liquidation feed): a cascade appears in public data
as OI FLUSH + PRICE SPIKE in the same window — open interest drops sharply
(positions force-closed) while price moves hard in one direction.

Data: Binance USDT-perp 5m open-interest history (API keeps ~30 days) + 5m
candles, 8 competition pairs. Cached to data/liq_scan/.

Event (pre-declared, mechanism-driven — NOT fitted):
    over the last 3 bars (15m):
      OI change <= -1.5%   AND   |price return| >= 2 x ATR(15m)
    direction = sign of the price move; cooldown = no new event inside the
    longest forward horizon.
A sensitivity grid around the thresholds is printed for context, with the
multiple-testing warning attached. Trust the pre-declared cell + halves.

Trade tested: fade the flush (buy a long-liquidation dump / sell a short
squeeze) at the event bar close, exit after 15m/30m/1h/2h/4h.
Costs: maker 0.04% RT (rest a post-only into the flush) and taker 0.12% RT.

    python run_liq_scan.py            # 30 days, 8 pairs
    python run_liq_scan.py --days 21
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import ccxt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, ".")

console = Console()

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "LTC/USDT:USDT",
]
MAKER_RT = 0.0004
TAKER_RT = 0.0012
CACHE = Path("data/liq_scan")
BAR_MS = 5 * 60 * 1000
HORIZONS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48}
OI_DROP = -0.015   # pre-declared
ATR_MULT = 2.0     # pre-declared
WINDOW = 3         # bars (15m)



# tokenized equity/commodity/ETF perps that pollute a "top by volume" crypto
# universe — market-hours microstructure fires fake cascade events at opens
NON_CRYPTO = {
    "SNDK", "SOXL", "SKHYNIX", "SKHY", "XAU", "XAG", "MU", "SPCX", "KORU",
    "AKE", "CL", "BZ", "DRAM", "US", "EWY", "LAB", "SAMSUNG", "QQQ", "BANK",
    "INTC", "ESPORTS", "MSTR", "HOME", "MRVL", "NVDA", "TSLA", "AAPL", "SPY",
    "COIN", "HOOD", "GOOGL", "AMZN", "META", "MSFT", "GLD", "USO", "TQQQ",
}


def top_perps(ex, n: int, crypto_only: bool = False) -> list[str]:
    """Top-n USDT perps by 24h quote volume (validation universe — the
    mechanism is not specific to the 8 competition pairs)."""
    tickers = ex.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if sym.endswith("/USDT:USDT") and t.get("quoteVolume"):
            if crypto_only and sym.split("/")[0] in NON_CRYPTO:
                continue
            rows.append((t["quoteVolume"], sym))
    rows.sort(reverse=True)
    return [s for _, s in rows[:n]]


def dedupe_clustered(events, window_ms=30 * 60 * 1000):
    """Market-wide flushes fire many symbols in the same minutes — those are
    ONE observation, not twenty. Keep the largest |move| per time cluster so
    t-stats aren't inflated by cross-sectional correlation."""
    events = sorted(events, key=lambda e: e["t"])
    out = []
    for e in events:
        if out and e["t"] - out[-1]["t"] < window_ms:
            if abs(e["move_atr"]) > abs(out[-1]["move_atr"]):
                out[-1] = e
        else:
            out.append(e)
    return out


def fetch_series(days: int, symbols: list[str]):
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    now = int(time.time() * 1000)
    start = now - days * 86400_000
    for sym in symbols:
        tag = sym.split("/")[0]
        cf = CACHE / f"{tag}_{days}d.json"
        if cf.exists() and now - json.loads(cf.read_text())["saved"] < 6 * 3600_000:
            d = json.loads(cf.read_text())
            out[sym] = (np.array(d["ts"]), np.array(d["oi"]), np.array(d["c"]),
                        np.array(d["h"]), np.array(d["l"]))
            console.print(f"  {tag}: cached ({len(d['ts'])} bars)")
            continue
        oi_rows = []
        cursor = start
        while cursor < now:
            try:
                chunk = ex.fetch_open_interest_history(
                    sym, timeframe="5m", since=cursor, limit=500
                )
            except ccxt.BaseError as e:
                console.print(f"[yellow]{tag} OI window: {str(e)[:80]}[/yellow]")
                chunk = []
            if not chunk:
                cursor += 500 * BAR_MS
                continue
            oi_rows.extend((r["timestamp"], r["openInterestAmount"]) for r in chunk)
            cursor = chunk[-1]["timestamp"] + BAR_MS
            time.sleep(ex.rateLimit / 1000)
        ohlcv = []
        cursor = start
        while cursor < now:
            chunk = ex.fetch_ohlcv(sym, "5m", since=cursor, limit=1000)
            if not chunk:
                break
            ohlcv.extend(chunk)
            cursor = chunk[-1][0] + BAR_MS
            time.sleep(ex.rateLimit / 1000)
        oi_map = dict(oi_rows)
        ts, oi, c, h, l = [], [], [], [], []
        for row in ohlcv:
            if row[0] in oi_map:
                ts.append(row[0]); oi.append(oi_map[row[0]])
                c.append(row[4]); h.append(row[2]); l.append(row[3])
        cf.write_text(json.dumps({"saved": now, "ts": ts, "oi": oi, "c": c, "h": h, "l": l}))
        out[sym] = tuple(np.array(x) for x in (ts, oi, c, h, l))
        console.print(f"  {tag}: {len(ts)} aligned 5m bars")
    return out


def atr(h, l, c, n=288):  # trailing 1 day of 5m bars
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    csum = np.concatenate([[0.0], np.cumsum(tr)])  # csum[k] = sum of first k TRs
    out = np.full(len(c), np.nan)
    out[n:] = (csum[n:] - csum[:-n]) / n  # ATR at bar i = mean TR of bars i-n+1..i
    return out


def find_events(ts, oi, c, h, l, oi_drop, atr_mult):
    a = atr(h, l, c)
    events = []
    max_h = max(HORIZONS.values())
    last_i = -10**9
    for i in range(WINDOW + 289, len(c) - max_h):
        if i - last_i <= max_h:
            continue
        doi = oi[i] / oi[i - WINDOW] - 1
        dpx = c[i] - c[i - WINDOW]
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        if doi <= oi_drop and abs(dpx) >= atr_mult * a[i] * WINDOW ** 0.5:
            side = -np.sign(dpx)  # fade the flush
            fwd = {k: side * np.log(c[i + n] / c[i]) for k, n in HORIZONS.items()}
            events.append({"i": i, "t": ts[i], "side": side, "doi": doi,
                           "move_atr": dpx / a[i], "fwd": fwd})
            last_i = i
    return events


def summarize(events, label):
    if not events:
        console.print(f"[yellow]{label}: no events[/yellow]")
        return
    events = sorted(events, key=lambda e: e["t"])
    half_t = events[len(events) // 2]["t"]
    tbl = Table(title=f"{label} — {len(events)} events (fade the flush)")
    for col in ["horizon", "mean fwd", "t-stat", "1st half", "2nd half",
                "win%", "net maker", "net taker"]:
        tbl.add_column(col, justify="right")
    for hz in HORIZONS:
        r = np.array([e["fwd"][hz] for e in events])
        r1 = np.array([e["fwd"][hz] for e in events if e["t"] < half_t])
        r2 = np.array([e["fwd"][hz] for e in events if e["t"] >= half_t])
        tstat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else 0
        nm, nt = r.mean() - MAKER_RT, r.mean() - TAKER_RT
        tbl.add_row(
            hz, f"{r.mean()*100:+.3f}%", f"{tstat:.1f}",
            f"{r1.mean()*100:+.3f}%" if len(r1) else "-",
            f"{r2.mean()*100:+.3f}%" if len(r2) else "-",
            f"{(r > 0).mean()*100:.0f}%",
            f"[{'green' if nm > 0 else 'red'}]{nm*100:+.3f}%[/]",
            f"[{'green' if nt > 0 else 'red'}]{nt*100:+.3f}%[/]",
        )
    console.print(tbl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--top", type=int, default=0,
                    help="scan top-N Binance perps by volume instead of the 8 competition pairs")
    ap.add_argument("--crypto-only", action="store_true",
                    help="exclude tokenized equity/commodity perps from the universe")
    a = ap.parse_args()
    if a.top:
        symbols = top_perps(ccxt.binanceusdm({"enableRateLimit": True}), a.top, a.crypto_only)
        console.print(f"[dim]universe: top {len(symbols)} perps by 24h volume"
                      f"{' (crypto only)' if a.crypto_only else ''}[/dim]")
    else:
        symbols = SYMBOLS
    console.print(Panel(f"[bold]Liquidation-cascade bounce scan — {a.days}d, 5m bars, {len(symbols)} pairs[/bold]\n"
                        f"pre-declared event: OI {OI_DROP*100:.1f}% in 15m AND |move| >= {ATR_MULT}xATR"))
    data = fetch_series(a.days, symbols)

    all_events = []
    per_sym = {}
    for sym, (ts, oi, c, h, l) in data.items():
        ev = find_events(ts, oi, c, h, l, OI_DROP, ATR_MULT)
        per_sym[sym] = ev
        all_events.extend(ev)
    console.print(f"\nevents per symbol: " +
                  ", ".join(f"{s.split('/')[0]}:{len(v)}" for s, v in per_sym.items()) + "\n")

    summarize(all_events, "PRE-DECLARED CELL (pooled, all symbols)")

    deduped = dedupe_clustered(all_events)
    summarize(deduped, f"DEDUPED (1 event per 30min cluster — honest t-stats)")

    longs = [e for e in deduped if e["side"] > 0]
    shorts = [e for e in deduped if e["side"] < 0]
    summarize(longs, "deduped long-liquidation dumps (we BUY the flush)")
    summarize(shorts, "deduped short-squeeze pumps (we SELL the flush)")

    console.print("\n[bold]sensitivity grid (context only — multiple testing!)[/bold]")
    tbl = Table()
    for col in ["oi_drop", "atr_mult", "events", "15m", "t", "net mkr 15m", "1h", "t", "net mkr 1h"]:
        tbl.add_column(col, justify="right")
    for od in (-0.01, -0.015, -0.02, -0.03):
        for am in (1.5, 2.0, 3.0):
            evs = []
            for sym, (ts, oi, c, h, l) in data.items():
                evs.extend(find_events(ts, oi, c, h, l, od, am))
            if len(evs) < 5:
                tbl.add_row(f"{od*100:.1f}%", f"{am}", str(len(evs)), *["-"] * 6)
                continue
            cells = [f"{od*100:.1f}%", f"{am}", str(len(evs))]
            for hz in ("15m", "1h"):
                r = np.array([e["fwd"][hz] for e in evs])
                tstat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if r.std() > 0 else 0
                nm = r.mean() - MAKER_RT
                cells += [f"{r.mean()*100:+.3f}%", f"{tstat:.1f}",
                          f"[{'green' if nm > 0 else 'red'}]{nm*100:+.3f}%[/]"]
            tbl.add_row(*cells)
    console.print(tbl)
    console.print(
        "[dim]Bar: pre-declared cell must be net-positive at maker cost with same sign in both "
        "halves AND survive on a SECOND non-overlapping period (rerun in a week) before any "
        "wiring. 30d of history is a screen, not proof. Grid cells are context, not cherry-picks. "
        "Caveat: OI proxy also fires on voluntary de-risking; live version should confirm with "
        "the forceOrder liquidation stream.[/dim]"
    )


if __name__ == "__main__":
    main()
