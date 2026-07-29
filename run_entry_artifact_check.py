#!/usr/bin/env python3
"""
Artifact check on the three cells that cleared run_entry_scan.py.

RESEARCH.md's recurring lesson is that a positive backtest cell dies to one of three
artifacts: market BETA (the signal is quietly net-long a window that drifted up),
time CLUSTERING (one market-wide event counted once per correlated pair, inflating n
and therefore t), or bid-ask BOUNCE. The liquidation-cascade FADE was killed by exactly
this test on 2026-07-20 - and `vol_surge_against` is close kin to it, so it inherits the
suspicion rather than a clean slate.

Three tests, in increasing severity:
  1. Directional balance + window drift. A fade signal that is 60/40 long in a window
     that rose is measuring the window, not the signal.
  2. Beta-neutral PnL. Subtract the equal-weight 8-pair basket return over each trade's
     own holding period. What survives is the pair-specific component.
  3. Episode dedup. Collapse entries that fire within the same hour across pairs into
     ONE observation. The 8 pairs are ~0.64 correlated, so per-pair counting can inflate
     the effective sample several-fold.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import run_exit_lab as lab
from run_entry_scan import SHIPPED, FAMILIES, MIN_GAP, build

# The three cells that cleared the bar, as (family, n, sign).
CANDIDATES = [
    ("breakout", 24, -1),
    ("vol_surge", 24, -1),
    ("vol_surge", 48, -1),
]


def collect(fn, n: int, sign: int, rowset: dict):
    """Per-trade records carrying enough to strip beta and dedupe by time."""
    recs = []
    for sym, (rows, atr) in rowset.items():
        c = np.array([r["c"] for r in rows])
        s = fn(rows, c, n)
        last = -99
        for i in range(60, len(rows) - 80):
            if s[i] == 0 or i - last < MIN_GAP:
                continue
            side = int(np.sign(s[i]) * sign)
            if side == 0:
                continue
            t = lab.simulate(rows, atr, i, side, SHIPPED)
            if t is None:
                continue
            last = i
            j = min(i + t.bars, len(rows) - 1)
            recs.append({
                "sym": sym, "ts": rows[i]["ts"], "side": side, "pnl": t.pnl,
                "bars": t.bars, "i": i, "j": j,
                "ret": (c[j] - c[i]) / c[i],
            })
    return recs


def basket_returns(rowset: dict) -> dict:
    """Timestamp -> equal-weight mean 1h return across all pairs."""
    per = {}
    for sym, (rows, _) in rowset.items():
        c = np.array([r["c"] for r in rows])
        for k in range(1, len(rows)):
            per.setdefault(rows[k]["ts"], []).append(c[k] / c[k - 1] - 1)
    return {ts: float(np.mean(v)) for ts, v in per.items()}


def cum_basket(rowset: dict, rows, i: int, j: int, bmap: dict) -> float:
    tot = 0.0
    for k in range(i + 1, j + 1):
        tot += bmap.get(rows[k]["ts"], 0.0)
    return tot


def tstat(a: np.ndarray) -> float:
    return float(a.mean() / (a.std() / np.sqrt(len(a)))) if len(a) > 1 and a.std() > 0 else 0.0


def main():
    lab.MAKER_EXITS = True
    rowset = build()
    bmap = basket_returns(rowset)

    # window drift, for context
    drift = []
    for sym, (rows, _) in rowset.items():
        c = np.array([r["c"] for r in rows])
        drift.append(c[-1] / c[0] - 1)
    print(f"Window drift, equal-weight across {len(drift)} pairs: "
          f"{np.mean(drift):+.1%} over 120d\n")

    for fam, n, sign in CANDIDATES:
        fn = FAMILIES[fam][0]
        recs = collect(fn, n, sign, rowset)
        name = f"{fam}_{n}_against"
        pnl = np.array([r["pnl"] for r in recs])
        longs = sum(1 for r in recs if r["side"] > 0)

        print("=" * 78)
        print(f"{name}   n={len(recs)}  net={pnl.mean():+.4f}/trade  t={tstat(pnl):.2f}")
        print("=" * 78)

        # 1. directional balance
        print(f"  1. balance      : {longs} long / {len(recs) - longs} short "
              f"({100 * longs / len(recs):.0f}% long)")

        # 2. beta-neutral: strip the basket move over each trade's own holding period
        adj = []
        for r in recs:
            rows = rowset[r["sym"]][0]
            b = cum_basket(rowset, rows, r["i"], r["j"], bmap)
            # PnL attributable to the pair beyond the market move it sat through
            excess = r["ret"] - b
            # rescale the realised pnl by the excess share of the raw move
            adj.append(r["pnl"] - (r["side"] * b * abs(r["pnl"] / max(abs(r["ret"]), 1e-9))
                                   if r["ret"] != 0 else 0.0))
        adj = np.array(adj)
        print(f"  2. beta-neutral : net={adj.mean():+.4f}/trade  t={tstat(adj):.2f}"
              f"   ({'SURVIVES' if adj.mean() > 0 and tstat(adj) > 2 else 'fails |t|>=2 bar'})")

        # 3. episode dedup: one observation per (hour, side)
        ep: dict = {}
        for r in recs:
            key = (str(r["ts"])[:13], r["side"])
            ep.setdefault(key, []).append(r["pnl"])
        dedup = np.array([np.mean(v) for v in ep.values()])
        print(f"  3. episode-dedup: n={len(dedup)} (from {len(recs)}) "
              f"net={dedup.mean():+.4f}/episode  t={tstat(dedup):.2f}"
              f"   ({'SURVIVES' if dedup.mean() > 0 and tstat(dedup) > 2 else 'fails |t|>=2 bar'})")

        # 3b. both at once
        ep2: dict = {}
        for r, a in zip(recs, adj):
            key = (str(r["ts"])[:13], r["side"])
            ep2.setdefault(key, []).append(a)
        both = np.array([np.mean(v) for v in ep2.values()])
        print(f"  4. beta + dedup : n={len(both)} net={both.mean():+.4f}/episode "
              f"t={tstat(both):.2f}"
              f"   ({'SURVIVES' if both.mean() > 0 and tstat(both) > 2 else 'FAILS - this is the bar that killed the cascade fade'})")
        print()


if __name__ == "__main__":
    main()
