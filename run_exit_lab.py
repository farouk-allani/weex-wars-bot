#!/usr/bin/env python3
"""
Exit-geometry lab — measures the exit rules in isolation from entry edge.

Why this exists
---------------
Live paper (2026-07-16..29, 20 trades) produced a payoff ratio of 0.63: winners
exited at +0.30%..+0.99% of price move, losers at -0.39%..-1.57%. No trade in 20
ever reached the pre-partial trail activation (1.2%). A 30% win rate needs payoff
> 2.0 to break even; 0.63 is structurally unrecoverable no matter how good entries
get. That is an exit-geometry question, and 20 trades cannot answer it.

Method
------
Entry edge and exit geometry are separable. Feed the SAME entries through
different exit rules and the difference is attributable to the exits alone. The
key run is `--entries random`: zero-alpha entries, where a perfect exit rule
returns exactly -costs. Any geometry that loses MORE than costs on random entries
is destroying value on its own, independent of whether the AI can pick direction.
That is a bug you can fix without solving alpha.

Conventions (identical across every geometry, so comparisons are fair)
  * 1h bars from data/cache. Extremes update from bar high/low (the live bot polls
    every 60s, so it does see intra-bar extremes); stop logic evaluates on close.
  * If a bar touches both the stop and the profit target, the STOP wins. Worst-case
    resolution of intra-bar path ambiguity — pessimistic, but applied uniformly.
  * Fees: maker entry, taker exit, plus slippage on the taker leg (config.yaml).

Usage
  python run_exit_lab.py --entries random --sweep
  python run_exit_lab.py --entries mr --sweep
  python run_exit_lab.py --entries random --geometry current --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

CACHE = Path("data/cache")
PAIRS = ["BTC", "ETH", "SOL", "DOGE", "ADA", "BNB", "LTC", "XRP"]

# From config.yaml: execution.maker_fee_rate / backtest.commission_rate + slippage_pct
MAKER_FEE = 0.0002
TAKER_FEE = 0.0006
SLIPPAGE = 0.0005

MAKER_EXITS = False  # set by --maker-exits

NOTIONAL = 300.0  # observed live: 240-450 notional on $1k equity

# The live bot sizes by RISK, not notional: amount = risk_amount / stop_distance
# (RiskManager.calculate_position_size), capped at max_position_pct of equity.
# Comparing stop WIDTHS under fixed notional is therefore meaningless - a wider stop
# would simply carry more dollars of risk and "win" by leverage alone. RISK_SIZING
# reproduces the real rule so stop width can be compared honestly.
RISK_SIZING = True
EQUITY = 1000.0
MAX_RISK_PER_TRADE = 0.012   # config.yaml risk.max_risk_per_trade
MAX_POSITION_PCT = 0.75      # config.yaml backtest.max_position_pct


# ---------------------------------------------------------------- data


def load_pair(sym: str) -> list[dict] | None:
    """Longest available 1h series for a pair."""
    files = sorted(
        CACHE.glob(f"{sym}_USDT_USDT_1h_*.json"),
        key=lambda p: int(p.stem.split("_")[-1].rstrip("d")),
    )
    if not files:
        return None
    rows = json.loads(files[-1].read_text())["rows"]
    return rows


def atr_series(rows: list[dict], period: int = 14) -> np.ndarray:
    h = np.array([r["h"] for r in rows])
    l = np.array([r["l"] for r in rows])
    c = np.array([r["c"] for r in rows])
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full(len(tr), np.nan)
    if len(tr) <= period:
        return out
    out[period] = tr[1 : period + 1].mean()
    for i in range(period + 1, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


# ---------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Geometry:
    name: str
    stop_atr: float = 1.2          # initial stop distance, in ATR
    target_atr: float = 2.4        # full take-profit, in ATR
    partial_enabled: bool = True
    partial_at_r: float = 1.0      # partial TP trigger, in R
    partial_frac: float = 0.5
    be_trigger_r: float = 1.0      # move stop to BE at this many R
    be_trigger_r_post: float = 0.7  # ...after a partial has been taken
    be_buffer_atr: float = 0.12    # where BE sits above entry, in ATR
    trail_act: float = 0.012       # trail arms at this profit fraction
    trail_act_post: float = 0.0066
    trail_pct: float = 0.006       # percent trail distance
    trail_pct_post: float = 0.0045
    chand_atr: float = 2.3         # chandelier distance, in ATR
    chand_atr_post: float = 1.61
    # The live code does trail = max(chandelier, pct_trail) for longs, i.e. it always
    # takes the TIGHTER of the two. Setting this False takes the looser, letting the
    # volatility-adaptive chandelier bind instead of the fixed percent.
    trail_takes_tighter: bool = True
    max_bars: int = 72             # force flat after 3 days


CURRENT = Geometry(name="current")


# ---------------------------------------------------------------- simulation


@dataclass
class Trade:
    pnl: float
    fees: float
    move_pct: float
    bars: int
    reason: str


def simulate(
    rows: list[dict],
    atr: np.ndarray,
    i: int,
    side: int,          # +1 long, -1 short
    g: Geometry,
) -> Trade | None:
    """Replay one position from bar i under geometry g."""
    a0 = atr[i]
    if not np.isfinite(a0) or a0 <= 0:
        return None
    entry = rows[i]["c"]
    if entry <= 0:
        return None

    risk = a0 * g.stop_atr
    if risk <= 0:
        return None

    if RISK_SIZING:
        # constant dollar risk at the stop, exactly like the live sizer
        size = (EQUITY * MAX_RISK_PER_TRADE) / risk
        size = min(size, EQUITY * MAX_POSITION_PCT / entry)
    else:
        size = NOTIONAL / entry
    initial_size = size

    stop = entry - side * risk
    target = entry + side * a0 * g.target_atr
    ptp = entry + side * risk * g.partial_at_r if g.partial_enabled else None
    if ptp is not None and side * (target - ptp) <= 0:
        ptp = None

    hi = entry
    lo = entry
    partial_taken = False
    banked = 0.0
    fees = size * entry * MAKER_FEE  # maker entry
    trail_stop = None

    def taker_exit(px: float, qty: float) -> float:
        return qty * px * (TAKER_FEE + SLIPPAGE)

    def limit_exit(px: float, qty: float) -> float:
        """
        Take-profit and partial legs rest at a known price, so they can be maker.
        Stops cannot: a stop crosses the book by construction. MAKER_EXITS only
        rebates the legs that are genuinely restable.
        """
        return qty * px * (MAKER_FEE if MAKER_EXITS else TAKER_FEE + SLIPPAGE)

    for k in range(i + 1, min(i + 1 + g.max_bars, len(rows))):
        bar = rows[k]
        h, l, c = bar["h"], bar["l"], bar["c"]
        a = atr[k] if np.isfinite(atr[k]) and atr[k] > 0 else a0

        hi = max(hi, h)
        lo = min(lo, l)

        adverse = l if side > 0 else h
        favour = h if side > 0 else l

        # --- stop first: worst-case resolution of intra-bar ordering
        if side * (adverse - stop) <= 0:
            px = stop
            pnl = side * (px - entry) * size + banked
            fees += taker_exit(px, size)
            in_profit = side * (stop - entry) > 0
            return Trade(
                pnl=pnl - fees,
                fees=fees,
                move_pct=side * (px - entry) / entry * 100,
                bars=k - i,
                reason="be_stop" if in_profit else "stop_loss",
            )

        # --- partial TP
        if ptp is not None and not partial_taken and side * (favour - ptp) >= 0:
            close_size = initial_size * g.partial_frac
            close_size = min(close_size, size)
            banked += side * (ptp - entry) * close_size
            fees += limit_exit(ptp, close_size)
            size -= close_size
            partial_taken = True
            be = entry + side * a * g.be_buffer_atr
            if side * (be - stop) > 0:
                stop = be
            if size <= 1e-12:
                return Trade(
                    pnl=banked - fees,
                    fees=fees,
                    move_pct=side * (ptp - entry) / entry * 100,
                    bars=k - i,
                    reason="partial_full",
                )

        # --- full take-profit
        if side * (favour - target) >= 0:
            px = target
            pnl = side * (px - entry) * size + banked
            fees += limit_exit(px, size)
            return Trade(
                pnl=pnl - fees,
                fees=fees,
                move_pct=side * (px - entry) / entry * 100,
                bars=k - i,
                reason="take_profit",
            )

        # --- adjust stops (mirrors RiskManager.adjust_stops)
        profit_pct = side * (c - entry) / entry
        cur_risk = abs(entry - stop) if stop > 0 else a * 1.5
        be_trig = g.be_trigger_r_post if partial_taken else g.be_trigger_r
        if side * (c - entry) >= cur_risk * be_trig:
            be = entry + side * a * g.be_buffer_atr
            if side * (be - stop) > 0:
                stop = be

        act = g.trail_act_post if partial_taken else g.trail_act
        cm = g.chand_atr_post if partial_taken else g.chand_atr
        td = g.trail_pct_post if partial_taken else g.trail_pct
        if profit_pct >= act:
            chandelier = (hi - a * cm) if side > 0 else (lo + a * cm)
            pct_trail = c * (1 - td) if side > 0 else c * (1 + td)
            if g.trail_takes_tighter:
                trail = max(chandelier, pct_trail) if side > 0 else min(chandelier, pct_trail)
            else:
                trail = min(chandelier, pct_trail) if side > 0 else max(chandelier, pct_trail)
            if trail_stop is None or side * (trail - trail_stop) > 0:
                trail_stop = trail
            if side * (trail_stop - stop) > 0:
                stop = trail_stop

    # --- timeout
    px = rows[min(i + g.max_bars, len(rows) - 1)]["c"]
    pnl = side * (px - entry) * size + banked
    fees += taker_exit(px, size)
    return Trade(
        pnl=pnl - fees,
        fees=fees,
        move_pct=side * (px - entry) / entry * 100,
        bars=min(g.max_bars, len(rows) - 1 - i),
        reason="timeout",
    )


# ---------------------------------------------------------------- entries


def entries_random(rows: list[dict], atr: np.ndarray, rng: random.Random, every: int = 8):
    """Zero-alpha control. A perfect exit rule returns exactly -costs here."""
    for i in range(20, len(rows) - 80, every):
        yield i, (1 if rng.random() < 0.5 else -1)


def make_oracle(p: float):
    """
    Entries with a KNOWN edge: take the correct side (by 24h forward return) with
    probability p. p=0.5 is random; p=0.55 is a strong-but-plausible real signal.
    This calibrates how much directional accuracy the costs actually demand, and
    which exit geometry best converts a given edge into money.
    """

    def gen(rows: list[dict], atr: np.ndarray, rng: random.Random, every: int = 8):
        c = np.array([r["c"] for r in rows])
        for i in range(20, len(rows) - 80, every):
            fwd = c[min(i + 24, len(c) - 1)] - c[i]
            truth = 1 if fwd > 0 else -1
            yield i, (truth if rng.random() < p else -truth)

    return gen


def entries_mr(rows: list[dict], atr: np.ndarray, rng: random.Random, every: int = 1):
    """Crude mean-reversion proxy: 20-bar z-score beyond +/-1, fade it."""
    c = np.array([r["c"] for r in rows])
    last = -99
    for i in range(40, len(rows) - 80):
        if i - last < 6:
            continue
        w = c[i - 20 : i]
        sd = w.std()
        if sd <= 0:
            continue
        z = (c[i] - w.mean()) / sd
        if z <= -1.0:
            last = i
            yield i, 1
        elif z >= 1.0:
            last = i
            yield i, -1


ENTRY_FNS = {
    "random": entries_random,
    "mr": entries_mr,
    "edge52": make_oracle(0.52),
    "edge55": make_oracle(0.55),
    "edge60": make_oracle(0.60),
}


# ---------------------------------------------------------------- reporting


def run(geom: Geometry, entry_kind: str, seed: int = 7) -> dict:
    rng = random.Random(seed)
    trades: list[Trade] = []
    for sym in PAIRS:
        rows = load_pair(sym)
        if not rows or len(rows) < 200:
            continue
        atr = atr_series(rows)
        for i, side in ENTRY_FNS[entry_kind](rows, atr, rng):
            t = simulate(rows, atr, i, side, geom)
            if t is not None:
                trades.append(t)

    if not trades:
        return {"name": geom.name, "n": 0}

    pnls = np.array([t.pnl for t in trades])
    fees = np.array([t.fees for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    aw = wins.mean() if len(wins) else 0.0
    al = losses.mean() if len(losses) else 0.0
    return {
        "name": geom.name,
        "n": len(trades),
        "net": pnls.sum(),
        "net_per_trade": pnls.mean(),
        "fees_per_trade": fees.mean(),
        "gross_per_trade": pnls.mean() + fees.mean(),
        "winrate": 100 * len(wins) / len(trades),
        "avg_win": aw,
        "avg_loss": al,
        "payoff": abs(aw / al) if al else float("inf"),
        "expectancy_r": pnls.mean(),
        "std": pnls.std(),
        "sharpe_per_trade": pnls.mean() / pnls.std() if pnls.std() else 0.0,
        "avg_bars": float(np.mean([t.bars for t in trades])),
        "reasons": {r: sum(1 for t in trades if t.reason == r) for r in
                    sorted({t.reason for t in trades})},
    }


def fmt(r: dict) -> str:
    if not r.get("n"):
        return f"{r['name']:<22} (no trades)"
    return (
        f"{r['name']:<22} n={r['n']:<5} net/trade={r['net_per_trade']:+7.4f} "
        f"gross/trade={r['gross_per_trade']:+7.4f} "
        f"WR={r['winrate']:4.1f}% payoff={r['payoff']:5.2f} "
        f"sharpe={r['sharpe_per_trade']:+6.3f} bars={r['avg_bars']:5.1f}"
    )


def build_sweep() -> list[Geometry]:
    g = []
    g.append(CURRENT)
    # The single-line change: stop always taking the tighter of chandelier/pct.
    g.append(replace(CURRENT, name="chandelier_binds", trail_takes_tighter=False))
    # Widen the percent trail instead, keeping max().
    for d in (0.010, 0.015, 0.020, 0.030):
        g.append(replace(CURRENT, name=f"trail_pct={d:.3f}",
                         trail_pct=d, trail_pct_post=d * 0.75))
    # No percent trail at all — pure chandelier.
    g.append(replace(CURRENT, name="chandelier_only",
                     trail_pct=0.99, trail_pct_post=0.99))
    # Give the BE stop real room instead of pinning it at entry.
    for b in (0.5, 1.0):
        g.append(replace(CURRENT, name=f"be_buffer={b}atr", be_buffer_atr=b,
                         trail_takes_tighter=False))
    # Turn the partial off (it is what arms the tight post-partial trail).
    g.append(replace(CURRENT, name="no_partial", partial_enabled=False))
    g.append(replace(CURRENT, name="no_partial+chand",
                     partial_enabled=False, trail_takes_tighter=False))
    # Pure bracket: stop + target, no trail, no partial, no BE.
    g.append(replace(CURRENT, name="pure_bracket", partial_enabled=False,
                     be_trigger_r=99, trail_act=99))
    # Wider target with the chandelier binding.
    for t in (3.0, 4.0):
        g.append(replace(CURRENT, name=f"target={t}atr+chand", target_atr=t,
                         trail_takes_tighter=False))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", choices=list(ENTRY_FNS), default="random")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--geometry", default="current")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fixed-notional", action="store_true",
                    help="size by notional instead of risk (breaks stop-width comparisons)")
    ap.add_argument("--maker-exits", action="store_true",
                    help="rest take-profit/partial legs as maker (stops stay taker)")
    args = ap.parse_args()

    global MAKER_EXITS, RISK_SIZING
    MAKER_EXITS = args.maker_exits
    RISK_SIZING = not args.fixed_notional

    geoms = build_sweep() if args.sweep else [CURRENT]

    print(f"\n=== exit lab | entries={args.entries} | "
          f"fees: maker {MAKER_FEE:.4%} in / taker {TAKER_FEE:.4%}+{SLIPPAGE:.4%} out ===")
    if args.entries == "random":
        print("random entries have ZERO alpha: a non-destructive exit rule scores "
              "-costs/trade and nothing worse.\n")

    results = [run(g, args.entries, args.seed) for g in geoms]
    base = results[0]
    for r in results:
        line = fmt(r)
        if r.get("n") and r is not base and base.get("n"):
            d = r["net_per_trade"] - base["net_per_trade"]
            line += f"  [{d:+.4f} vs current]"
        print(line)

    if args.verbose:
        print()
        for r, g in zip(results, geoms):
            if r.get("n"):
                print(f"{r['name']}: {r['reasons']}")


if __name__ == "__main__":
    main()
