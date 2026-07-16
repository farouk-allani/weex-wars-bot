"""Funding-carry RV on WEEX's OWN funding prints — the missing validation gate.

run_rv_scan.py found the one net-positive-after-maker-cost edge: long the 2
lowest-funding coins, short the 2 highest, hold 3-7d. BUT its funding leg came
from BINANCE (src/data/fetcher.fetch_funding_map uses ccxt.binance). We trade
on WEEX, whose funding prints come from a different crowd. This tool re-runs
the exact strategy with WEEX funding history (public, 365d back, 7-day windows
per call, cached locally) so the carry leg is measured where it will be paid.

Price leg stays Binance daily closes (same as the original scan — apples to
apples: the ONLY thing that changes is the funding source). Cross-venue daily
closes are near-identical; funding is the venue-specific part.

Modes:
    python run_carry_weex.py                  # backtest on WEEX funding, 360d
    python run_carry_weex.py --days 180
    python run_carry_weex.py --live           # snapshot now: basket, ex-ante
                                              # carry vs cost, would-we-trade,
                                              # appended to data/carry_forward.jsonl
    python run_carry_weex.py --paper          # FORWARD paper book of the gated
                                              # 3d policy: enters/holds/closes a
                                              # virtual basket at live marks with
                                              # exact funding prints; state in
                                              # data/carry_paper_state.json,
                                              # closed trades appended to
                                              # data/carry_paper_trades.jsonl.
                                              # Run 1-3x/day (each run advances it).

Verdict bar (same as always): net-positive at maker cost over the full period
AND in the out-of-sample second half, with the ex-ante gate honest. Then
forward paper-tracking before real size.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()  # WEEX certs fail Python's bundled CAs on this box
except ImportError:
    pass

import ccxt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, ".")
from src.data.fetcher import fetch_ohlcv

console = Console()

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "LTC/USDT:USDT",
]
MAKER_RT = 0.0010  # 0.10% round trip per leg (same bar as run_rv_scan)
MARKET_RT = 0.0022
CACHE_DIR = Path("data/weex_funding")
FWD_FILE = Path("data/carry_forward.jsonl")
EIGHT_H_MS = 8 * 3600 * 1000


def weex_exchange():
    return ccxt.weex({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def download_weex_funding(ex, symbol: str, days: int) -> list[dict]:
    """Paginate 7-day windows (API limit). Cached to disk, refreshed tail only."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol.replace('/', '_').replace(':', '_')}.json"
    rows = []
    if cache.exists():
        rows = json.loads(cache.read_text())
    have_until = max((r["ts"] for r in rows), default=None)
    have_from = min((r["ts"] for r in rows), default=None)

    now = int(time.time() * 1000)
    start = now - days * 86400_000
    if have_from is not None and have_from > start + 7 * 86400_000:
        rows = []  # cache doesn't reach far enough back — refetch the range
        have_until = None
    # resume after cache tail; -1 window overlap to avoid gaps
    cursor = max(start, (have_until - EIGHT_H_MS) if have_until else start)
    seen = {r["ts"] for r in rows}
    while cursor < now:
        end = min(cursor + 7 * 86400_000 - 1, now)
        try:
            h = ex.fetch_funding_rate_history(
                symbol, since=cursor, limit=200, params={"endTime": end}
            )
        except ccxt.BaseError as e:
            console.print(f"[yellow]{symbol} window {cursor}: {str(e)[:90]} — skipping[/yellow]")
            h = []
        for r in h:
            ts = r["timestamp"]
            if ts not in seen:
                rows.append({"ts": ts, "rate": r["fundingRate"]})
                seen.add(ts)
        cursor = end + 1
        time.sleep(ex.rateLimit / 1000)
    rows.sort(key=lambda r: r["ts"])
    cache.write_text(json.dumps(rows))
    return rows


def build_matrices(days: int):
    """Aligned daily matrices: funding sum per day (3 prints), daily closes."""
    ex = weex_exchange()
    ex.load_markets()
    funding = {}
    for sym in SYMBOLS:
        rows = download_weex_funding(ex, sym, days)
        console.print(f"  {sym}: {len(rows)} funding prints "
                      f"({datetime.fromtimestamp(rows[0]['ts']/1000, tz=timezone.utc).date()} -> "
                      f"{datetime.fromtimestamp(rows[-1]['ts']/1000, tz=timezone.utc).date()})")
        funding[sym] = rows

    closes = {}
    for sym in SYMBOLS:
        candles = fetch_ohlcv(sym, "1d", days=days + 5)
        closes[sym] = {c.timestamp.date(): c.close for c in candles}

    # common daily grid
    all_days = sorted(
        {datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).date()
         for rows in funding.values() for r in rows}
    )
    grid = [d for d in all_days if all(d in closes[s] for s in SYMBOLS)]

    n, m = len(grid), len(SYMBOLS)
    F = np.full((n, m), np.nan)  # sum of that day's funding prints (what you pay/receive that day)
    P = np.full((n, m), np.nan)  # daily close
    for j, sym in enumerate(SYMBOLS):
        per_day = {}
        for r in funding[sym]:
            d = datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).date()
            per_day.setdefault(d, []).append(r["rate"])
        for i, d in enumerate(grid):
            if d in per_day:
                F[i, j] = sum(per_day[d])
            P[i, j] = closes[sym][d]
    return grid, F, P


def simulate(grid, F, P, hold: int, gate: bool):
    """Rebalance every `hold` days: long 2 lowest / short 2 highest trailing-24h
    funding. Carry accrues from NEXT day's prints (no lookahead). If gate=True,
    skip rebalances whose ex-ante carry differential < amortized maker cost."""
    n, m = F.shape
    rets, carries, prices, dates, traded = [], [], [], [], []
    w_prev = np.zeros(m)
    for i in range(1, n - hold, hold):
        sig = F[i]  # today's prints = trailing signal, positions enter at close i
        if np.isnan(sig).any():
            continue
        order = np.argsort(sig)
        w = np.zeros(m)
        w[order[:2]] = 0.25   # long lowest funding
        w[order[-2:]] = -0.25  # short highest funding
        # ex-ante: expected carry over hold if rates persist = -(w · sig) × hold
        exp_carry = float(-(w @ sig) * hold)
        turn = np.abs(w - w_prev).sum() / 2  # fraction of book replaced
        cost = turn * MAKER_RT
        if gate and exp_carry < cost:
            w_prev = np.zeros(m)  # flat this block
            continue
        carry = float(-(w * np.nansum(F[i + 1 : i + 1 + hold], axis=0)).sum())
        price = float((w * np.log(P[i + hold] / P[i])).sum())
        rets.append(carry + price - cost)
        carries.append(carry)
        prices.append(price)
        dates.append(grid[i])
        traded.append(exp_carry)
        w_prev = w
    return np.array(rets), np.array(carries), np.array(prices), dates, traded


def report(days: int):
    console.print(Panel(f"[bold]WEEX-native funding-carry validation — {days}d, 8 pairs[/bold]"))
    grid, F, P = build_matrices(days)
    console.print(f"\naligned days: {len(grid)} ({grid[0]} -> {grid[-1]})\n")

    for hold in (3, 7):
        for gate in (False, True):
            r, c, p, dates, exp = simulate(grid, F, P, hold, gate)
            if len(r) == 0:
                console.print(f"hold {hold}d gate={gate}: no trades")
                continue
            half = len(r) // 2
            lab = f"hold {hold}d {'GATED' if gate else 'always-on'}"
            tot, h1, h2 = r.sum(), r[:half].sum(), r[half:].sum()
            shp = r.mean() / r.std(ddof=1) * np.sqrt(365 / hold) if r.std() > 0 else 0
            eq = np.cumsum(r)
            dd = float((np.maximum.accumulate(eq) - eq).max())
            tbl = Table(title=f"{lab}: {len(r)} rebalances")
            for col in ["net total", "1st half", "2nd half (OOS)", "carry leg", "price leg", "ann.Sharpe", "maxDD"]:
                tbl.add_column(col, justify="right")
            tbl.add_row(
                f"[{'green' if tot > 0 else 'red'}]{tot*100:+.2f}%[/]",
                f"{h1*100:+.2f}%",
                f"[{'green' if h2 > 0 else 'red'}]{h2*100:+.2f}%[/]",
                f"{c.sum()*100:+.2f}%", f"{p.sum()*100:+.2f}%", f"{shp:.2f}", f"{dd*100:.2f}%",
            )
            console.print(tbl)
    console.print(
        "[dim]Same bar as ever: green needs net>0 full period AND 2nd half at maker cost. "
        "Carry leg is contractual; price leg is the unproven bonus. "
        "Costs: maker 0.10% RT per leg on turnover. Gate = trade only when ex-ante carry "
        "over the hold exceeds amortized cost (the deployable policy).[/dim]"
    )


def live_snapshot():
    ex = weex_exchange()
    ex.load_markets()
    rates = {}
    for sym in SYMBOLS:
        fr = ex.fetch_funding_rate(sym)
        rates[sym] = {
            "rate": fr.get("fundingRate"),
            "next": fr.get("fundingDatetime"),
            "mark": fr.get("markPrice") or fr.get("indexPrice"),
        }
        time.sleep(ex.rateLimit / 1000)

    order = sorted(SYMBOLS, key=lambda s: rates[s]["rate"])
    longs, shorts = order[:2], order[-2:]
    # expected carry if rates persist: 3 prints/day, book weight 0.25/leg
    diff_8h = (sum(rates[s]["rate"] for s in shorts) - sum(rates[s]["rate"] for s in longs)) * 0.25
    exp_3d, exp_7d = diff_8h * 9, diff_8h * 21
    cost = MAKER_RT  # full book turnover, amortized over the hold
    verdict = "TRADE 7d" if exp_7d > cost else ("TRADE 3d" if exp_3d > cost else "STAND ASIDE")

    tbl = Table(title=f"WEEX funding now — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    for col in ["symbol", "funding/8h", "role"]:
        tbl.add_column(col, justify="right")
    for s in order:
        role = "LONG" if s in longs else ("SHORT" if s in shorts else "")
        tbl.add_row(s.split("/")[0], f"{rates[s]['rate']*100:+.4f}%", role)
    console.print(tbl)
    console.print(
        f"ex-ante carry: 3d hold [bold]{exp_3d*100:+.4f}%[/bold], "
        f"7d hold [bold]{exp_7d*100:+.4f}%[/bold] "
        f"vs maker cost {cost*100:.2f}% -> [bold]{verdict}[/bold]"
    )

    FWD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FWD_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "t": datetime.now(timezone.utc).isoformat(),
            "rates": {s: rates[s]["rate"] for s in SYMBOLS},
            "marks": {s: rates[s]["mark"] for s in SYMBOLS},
            "longs": longs, "shorts": shorts,
            "exp_carry_3d": exp_3d, "exp_carry_7d": exp_7d, "cost": cost, "verdict": verdict,
        }) + "\n")
    console.print(f"[dim]snapshot appended to {FWD_FILE} (forward track record)[/dim]")


PAPER_STATE = Path("data/carry_paper_state.json")
PAPER_TRADES = Path("data/carry_paper_trades.jsonl")
HOLD_DAYS = 3  # the one cell that passed OOS (gated 3d)


def paper_step():
    """Advance the forward paper book one step. Idempotent between funding
    prints — safe to run any number of times per day."""
    ex = weex_exchange()
    ex.load_markets()
    now = datetime.now(timezone.utc)
    state = json.loads(PAPER_STATE.read_text()) if PAPER_STATE.exists() else {"equity": 1.0, "book": None}

    marks, rates = {}, {}
    for sym in SYMBOLS:
        fr = ex.fetch_funding_rate(sym)
        rates[sym] = fr.get("fundingRate")
        marks[sym] = fr.get("markPrice") or fr.get("indexPrice")
        time.sleep(ex.rateLimit / 1000)

    book = state.get("book")
    if book:
        opened = datetime.fromisoformat(book["opened"])
        held_days = (now - opened).total_seconds() / 86400
        # mark to market
        upnl = sum(w * np.log(marks[s] / book["entry_marks"][s]) for s, w in book["weights"].items())
        console.print(f"open book (day {held_days:.1f}/{HOLD_DAYS}): price uPnL {upnl*100:+.3f}%")
        if held_days >= HOLD_DAYS:
            since = int(opened.timestamp() * 1000)
            carry = 0.0
            for s, w in book["weights"].items():
                hist = ex.fetch_funding_rate_history(s, since=since, limit=200,
                                                     params={"endTime": int(now.timestamp() * 1000)})
                carry += -w * sum(h["fundingRate"] for h in hist)
                time.sleep(ex.rateLimit / 1000)
            cost = MAKER_RT  # round trip on the full book
            net = upnl + carry - cost
            state["equity"] *= 1 + net
            trade = {
                "opened": book["opened"], "closed": now.isoformat(),
                "weights": book["weights"], "price_pnl": upnl, "carry_pnl": carry,
                "cost": cost, "net": net, "equity": state["equity"],
            }
            PAPER_TRADES.parent.mkdir(parents=True, exist_ok=True)
            with open(PAPER_TRADES, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade) + "\n")
            state["book"] = None
            console.print(f"[bold]CLOSED: price {upnl*100:+.3f}% carry {carry*100:+.3f}% "
                          f"cost -{cost*100:.2f}% -> net {net*100:+.3f}% | equity {state['equity']:.4f}[/bold]")

    if state.get("book") is None:
        order = sorted(SYMBOLS, key=lambda s: rates[s])
        longs, shorts = order[:2], order[-2:]
        diff_8h = (sum(rates[s] for s in shorts) - sum(rates[s] for s in longs)) * 0.25
        exp = diff_8h * HOLD_DAYS * 3
        if exp > MAKER_RT:
            weights = {**{s: 0.25 for s in longs}, **{s: -0.25 for s in shorts}}
            state["book"] = {
                "opened": now.isoformat(), "weights": weights,
                "entry_marks": {s: marks[s] for s in weights},
                "exp_carry": exp,
            }
            console.print(f"[bold green]OPENED: long {[s.split('/')[0] for s in longs]} "
                          f"short {[s.split('/')[0] for s in shorts]} "
                          f"ex-ante carry {exp*100:+.3f}% > cost {MAKER_RT*100:.2f}%[/bold green]")
        else:
            console.print(f"gate says stand aside (ex-ante {exp*100:+.3f}% <= cost {MAKER_RT*100:.2f}%)")

    PAPER_STATE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE.write_text(json.dumps(state, indent=1))
    console.print(f"[dim]paper equity: {state['equity']:.4f} | state -> {PAPER_STATE}[/dim]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=360)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    if a.live:
        live_snapshot()
    elif a.paper:
        paper_step()
    else:
        report(a.days)
