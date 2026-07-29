#!/usr/bin/env python3
"""
Entry-signal scan, judged in MONEY under the shipped (fixed) exit geometry.

Why this is not re-slicing old data
-----------------------------------
Every entry hypothesis on the RESEARCH.md scoreboard was judged against an exit
geometry that demanded 62% directional accuracy to break even. That bar is now 54.8%
(see RESEARCH.md 2b). A signal that was correctly rejected as "edge < cost" under the
old geometry is not necessarily rejected under the new one, because the cost side of
that comparison moved. Re-running the candidates against the corrected bar is a
different question, asked once.

Discipline (the repo's own method, and the reason most of the scoreboard is DEAD)
---------------------------------------------------------------------------------
  * The candidate grid below is PRE-DECLARED in full and every cell is reported,
    winners and losers, so the multiple-testing burden is visible rather than hidden.
  * The bar is not IC and not a t-stat: it is NET MONEY per trade after maker-in /
    taker-out costs, POSITIVE IN BOTH out-of-sample halves independently.
  * Signals are evaluated through lab.simulate() - the same unbiased engine whose
    zero-alpha control returns exactly -costs - so nothing here can be flattered by
    an optimistic fill model.
  * Anything that passes is a CANDIDATE FOR FORWARD TESTING, not a discovery. The
    scoreboard is full of things that passed a backtest and died forward.

Usage
  python run_entry_scan.py
  python run_entry_scan.py --min-trades 80
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

import run_exit_lab as lab

# The geometry actually deployed (config.yaml + trader.py, 2026-07-29).
SHIPPED = replace(
    lab.CURRENT, name="shipped", partial_enabled=False, stop_atr=1.8,
    target_atr=2.6, be_trigger_r=1.75, be_buffer_atr=0.12,
    trail_act=0.012, trail_pct=0.02, chand_atr=2.3, trail_takes_tighter=False,
)


# ------------------------------------------------------------------ helpers


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru, rd = ema(up, n), ema(dn, n)
    return 100 - 100 / (1 + ru / np.maximum(rd, 1e-12))


# ------------------------------------------------------------------ signals
# Each returns an array of -1 / 0 / +1 per bar: the side to enter, 0 = stand aside.
# `sign` is applied by the caller so momentum and its reversal are one declared cell
# each, not a free extra degree of freedom picked after seeing results.


def sig_momentum(rows, c, n):
    out = np.zeros(len(c))
    r = np.zeros(len(c))
    r[n:] = c[n:] / c[:-n] - 1
    out[n:] = np.sign(r[n:])
    return out


def sig_zscore(rows, c, n):
    out = np.zeros(len(c))
    for i in range(n, len(c)):
        w = c[i - n:i]
        sd = w.std()
        if sd <= 0:
            continue
        z = (c[i] - w.mean()) / sd
        if abs(z) >= 1.0:
            out[i] = -np.sign(z)  # fade
    return out


def sig_breakout(rows, c, n):
    h = np.array([r["h"] for r in rows])
    l = np.array([r["l"] for r in rows])
    out = np.zeros(len(c))
    for i in range(n, len(c)):
        if c[i] >= h[i - n:i].max():
            out[i] = 1
        elif c[i] <= l[i - n:i].min():
            out[i] = -1
    return out


def sig_ema_cross(rows, c, n):
    f, s = ema(c, n), ema(c, n * 3)
    return np.sign(f - s)


def sig_rsi_extreme(rows, c, n):
    r = rsi(c, n)
    out = np.zeros(len(c))
    out[r <= 30] = 1
    out[r >= 70] = -1
    return out


def sig_vol_surge(rows, c, n):
    v = np.array([r["v"] for r in rows])
    out = np.zeros(len(c))
    for i in range(n, len(c)):
        m = v[i - n:i].mean()
        if m > 0 and v[i] > 2.0 * m:
            out[i] = np.sign(c[i] - rows[i]["o"])
    return out


# PRE-DECLARED GRID. Both signs of every family are declared up front, so reporting
# the better-performing sign afterwards is not a hidden choice.
FAMILIES = {
    "momentum": (sig_momentum, [6, 12, 24, 48]),
    "zscore_fade": (sig_zscore, [20, 50]),
    "breakout": (sig_breakout, [12, 24, 48]),
    "ema_cross": (sig_ema_cross, [8, 16]),
    "rsi_extreme": (sig_rsi_extreme, [10, 14]),
    "vol_surge": (sig_vol_surge, [24, 48]),
}
SIGNS = {"with": 1, "against": -1}
MIN_GAP = 6  # bars between entries on one pair, so one move is not counted many times


# ------------------------------------------------------------------ evaluation


def evaluate(fn, n: int, sign: int, rowset: dict) -> dict:
    pnls, correct = [], []
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
            pnls.append(t.pnl)
            fwd = c[min(i + 24, len(c) - 1)] - c[i]
            correct.append(1.0 if np.sign(fwd) == side else 0.0)
    if not pnls:
        return {"n": 0}
    a = np.array(pnls)
    return {
        "n": len(a),
        "net": a.mean(),
        "acc": float(np.mean(correct)) * 100,
        "t": float(a.mean() / (a.std() / np.sqrt(len(a)))) if a.std() > 0 else 0.0,
    }


def build(slice_: str | None = None) -> dict:
    out = {}
    for sym in lab.PAIRS:
        rows = lab.load_pair(sym)
        if not rows or len(rows) < 300:
            continue
        if slice_ == "first":
            rows = rows[: len(rows) // 2]
        elif slice_ == "second":
            rows = rows[len(rows) // 2:]
        out[sym] = (rows, lab.atr_series(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=60)
    args = ap.parse_args()

    lab.MAKER_EXITS = True
    full, h1, h2 = build(), build("first"), build("second")

    print("=" * 96)
    print("ENTRY SCAN - net $/trade under the SHIPPED exit geometry, maker TP legs, "
          "risk-sized")
    print("bar: net > 0 on the full sample AND > 0 in BOTH halves independently")
    print("=" * 96)
    print(f"{'signal':<26}{'n':>6}{'acc%':>7}{'net/trade':>11}{'t':>7}"
          f"{'half1':>9}{'half2':>9}   verdict")

    cells, passes = 0, []
    for fam, (fn, ns) in FAMILIES.items():
        for n in ns:
            for sname, sgn in SIGNS.items():
                cells += 1
                r = evaluate(fn, n, sgn, full)
                if r["n"] < args.min_trades:
                    print(f"{fam + f'_{n}_' + sname:<26}{r['n']:>6}"
                          f"{'':>7}{'':>11}{'':>7}{'':>9}{'':>9}   too few trades")
                    continue
                a = evaluate(fn, n, sgn, h1)
                b = evaluate(fn, n, sgn, h2)
                ok = r["net"] > 0 and a.get("net", -1) > 0 and b.get("net", -1) > 0
                verdict = "** PASSES BAR **" if ok else ("dead" if r["net"] <= 0 else "one half fails")
                if ok:
                    passes.append((f"{fam}_{n}_{sname}", r, a, b))
                print(f"{fam + f'_{n}_' + sname:<26}{r['n']:>6}{r['acc']:>7.1f}"
                      f"{r['net']:>+11.4f}{r['t']:>7.2f}"
                      f"{a.get('net', 0):>+9.3f}{b.get('net', 0):>+9.3f}   {verdict}")

    print("\n" + "=" * 96)
    print(f"{cells} cells tested. At a 5% false-positive rate, ~{cells * 0.05:.1f} cells "
          f"would clear a one-sided test by chance;\nrequiring BOTH halves positive "
          f"cuts that to ~{cells * 0.05 * 0.5:.1f}. Treat any pass accordingly.")
    if passes:
        print(f"\n{len(passes)} cell(s) cleared the bar:")
        for name, r, a, b in passes:
            print(f"  {name}: net {r['net']:+.4f}/trade, acc {r['acc']:.1f}%, "
                  f"t={r['t']:.2f}, halves {a['net']:+.3f} / {b['net']:+.3f}")
        print("\nThese are CANDIDATES FOR FORWARD TESTING, not results. The scoreboard "
              "is full of\ncells that passed a backtest and died on fresh data.")
    else:
        print("\nNothing cleared the bar. The exit geometry was not the only thing "
              "standing between\nthis bot and profitability - there is still no entry "
              "edge in these families at 1h.")


if __name__ == "__main__":
    main()
