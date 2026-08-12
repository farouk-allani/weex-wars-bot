#!/usr/bin/env python3
"""
Was the whole search run inside a box? — stop width x HOLD HORIZON, jointly.

Why this exists
---------------
Every entry hypothesis in RESEARCH.md died to the same sentence: edge < cost. But
"cost" is not a constant — it is `c/s`, the round-trip fee as a fraction of the stop
distance. Widen the stop and the same fee is amortised over a bigger R. That is the
lever run_stop_width.py found, and it stopped at 2.5 ATR because 3.0 scored worse.

Two artifacts made 3.0 look bad, and both bias AGAINST the wide/long end:

  1. `run_stop_width` sweeps `stop_atr` via `replace(geom, stop_atr=w)` and leaves
     `target_atr` at its 2.4 default. So the 3.0 row was tested at a reward:risk of
     0.8 — the target sitting NEARER than the stop. The live bot does not do this:
     `AITrader.to_signal` scales the target with the stop so the model's R:R
     survives. The sweep was measuring R:R decay, not stop width.

  2. `make_oracle(p)` defines "the correct side" as the sign of the **24-hour**
     forward return. Any geometry held longer than 24h is therefore being scored on
     information that has already expired — a longer horizon looks worse by
     construction, whatever the market does.

Fix both and the question becomes answerable: holding stop width and R:R constant,
does the geometry keep improving as the horizon lengthens, and where does it stop?

Method
------
  * target_atr = RR * stop_atr, so reward:risk is CONSTANT across every cell.
  * the oracle's lookahead is matched to the cell's horizon, so each geometry is
    judged on information of its own timescale.
  * objective is BREAK-EVEN ACCURACY (RESEARCH §2b's yardstick): the directional
    hit rate at which net PnL crosses zero. Lower is strictly better, and it is
    comparable across geometries in a way that raw PnL is not.
  * `every` scales with the horizon so long-hold cells are not just the same trade
    resampled 30 times; overlapping samples would understate the error.
  * the p=0.50 column is the zero-alpha control. A geometry that loses materially
    more than costs there is destroying value on its own.

This does NOT test an edge and cannot produce one. It asks a narrower question:
what accuracy would an edge need, and is the current 1h/3-day box the cheapest
place to need it?

Usage
  python run_horizon_scan.py
  python run_horizon_scan.py --rr 2.0 --oos
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

import run_exit_lab as lab
from run_exit_lab import Geometry, run

# `lab.CURRENT` is Geometry()'s DEFAULTS, which are the pre-§2b exit rules:
# be_trigger_r 1.0, a fixed 0.6% trail, and trail=max() taking the TIGHTER of
# chandelier/percent. config.yaml has shipped 1.75 / 0.02 / looser since d72a6c1.
# The difference is not cosmetic here: a 0.6% trail that always wins closes a
# winner almost immediately, which caps every long-horizon cell before it can
# resolve and would make "long horizons are bad" a property of the harness.
# Anything measured against horizon MUST use the shipped rules.
def shipped_geometry() -> Geometry:
    return replace(
        lab.CURRENT,
        name="shipped",
        partial_enabled=False,       # risk.partial_tp_enabled: false
        be_trigger_r=1.75,           # risk.be_trigger_r
        trail_pct=0.02,              # risk.trailing_stop_distance
        trail_act=0.012,             # risk.trailing_stop_activation
        chand_atr=2.3,               # risk.chandelier_atr_mult
        trail_takes_tighter=False,   # §2b fix: take the LOOSER candidate
    )


# Stop widths in ATR. Extends past run_stop_width's 3.0 ceiling because that
# ceiling is one of the things under test.
WIDTHS = [1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
# Hold horizons in 1h bars. 72 (3 days) is the shipped value.
HORIZONS = [24, 72, 168, 336]
# Accuracies used to locate the break-even point by linear fit.
PROBES = [0.50, 0.54, 0.58, 0.62]
SEEDS = [7, 11, 23]


def horizon_oracle(p: float, lookahead: int):
    """`make_oracle`, but the truth is the forward return over `lookahead` bars.

    The stock oracle hardcodes 24. Judging a 14-day hold on the sign of the next
    24 hours measures how fast the signal decays, not how well the geometry
    converts it — and it makes every long horizon look bad for free.
    """

    def gen(rows, atr, rng, every: int = 8):
        c = np.array([r["c"] for r in rows])
        # Never sample so densely that the same move is counted many times over.
        step = max(every, lookahead // 4)
        for i in range(20, len(rows) - lookahead - 20, step):
            fwd = c[min(i + lookahead, len(c) - 1)] - c[i]
            truth = 1 if fwd > 0 else -1
            yield i, (truth if rng.random() < p else -truth)

    return gen


def net_at(g: Geometry, p: float, lookahead: int) -> tuple[float, int]:
    key = f"_h{lookahead}_{int(p * 100)}"
    lab.ENTRY_FNS[key] = horizon_oracle(p, lookahead)
    vals, n = [], 0
    for s in SEEDS:
        r = run(g, key, seed=s)
        if r.get("n"):
            vals.append(r["net_per_trade"])
            n = r["n"]
    lab.ENTRY_FNS.pop(key, None)
    return (float(np.mean(vals)) if vals else float("nan"), n)


def breakeven_accuracy(g: Geometry, lookahead: int) -> tuple[float, float, int]:
    """Accuracy at which net/trade crosses zero, plus the zero-alpha cost."""
    nets, n_last = [], 0
    for p in PROBES:
        v, n = net_at(g, p, lookahead)
        nets.append(v)
        n_last = n or n_last
    nets_arr = np.array(nets)
    if not np.all(np.isfinite(nets_arr)):
        return float("nan"), float("nan"), n_last
    slope, intercept = np.polyfit(PROBES, nets_arr, 1)
    if slope <= 0:
        return float("nan"), nets[0], n_last
    return float(-intercept / slope), float(nets[0]), n_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, default=2.0,
                    help="reward:risk held CONSTANT across cells (target = rr * stop)")
    ap.add_argument("--oos", action="store_true", help="also split each cell in halves")
    args = ap.parse_args()

    lab.MAKER_EXITS = True
    lab.RISK_SIZING = True

    print("=" * 84)
    print(f"BREAK-EVEN ACCURACY  |  stop width x hold horizon  |  R:R fixed at {args.rr}")
    print("=" * 84)
    print("Lower is better. This is the directional hit rate the geometry DEMANDS.")
    print("The shipped config is stop 2.0 ATR / horizon 72 bars.\n")

    header = f"{'stop_atr':>9}" + "".join(f"{h:>13}" for h in HORIZONS)
    print(header)
    print(f"{'':>9}" + "".join(f"{'(' + str(h // 24) + 'd)':>13}" for h in HORIZONS))

    grid: dict[tuple[float, int], float] = {}
    costs: dict[tuple[float, int], float] = {}
    ns: dict[tuple[float, int], int] = {}
    for w in WIDTHS:
        cells = []
        for h in HORIZONS:
            g = replace(shipped_geometry(), name=f"s{w}h{h}", stop_atr=w,
                        target_atr=args.rr * w, max_bars=h)
            be, cost, n = breakeven_accuracy(g, h)
            grid[(w, h)] = be
            costs[(w, h)] = cost
            ns[(w, h)] = n
            cells.append(f"{be * 100:>8.1f}% n{n:<4d}" if np.isfinite(be) else f"{'n/a':>13}")
        print(f"{w:>9.1f}" + "".join(cells))

    print("\n" + "-" * 84)
    print("ZERO-ALPHA CONTROL: net $/trade at p=0.50 (pure cost; less negative = cheaper)")
    print("-" * 84)
    print(header)
    for w in WIDTHS:
        print(f"{w:>9.1f}" + "".join(f"{costs[(w, h)]:>12.4f}" for h in HORIZONS))

    finite = {k: v for k, v in grid.items() if np.isfinite(v)}
    if finite:
        best = min(finite, key=finite.get)
        base = grid.get((2.0, 72), float("nan"))
        print("\n" + "=" * 84)
        print(f"shipped  stop 2.0 / 72 bars (3d):  {base * 100:.1f}% accuracy needed"
              if np.isfinite(base) else "shipped cell did not resolve")
        print(f"best     stop {best[0]} / {best[1]} bars ({best[1] // 24}d):  "
              f"{finite[best] * 100:.1f}%")
        if np.isfinite(base):
            print(f"\ndifference: {(base - finite[best]) * 100:.1f} accuracy points")
            print("A LOWER bar does not create edge. It changes how much edge would be")
            print("needed to pay for itself — which is the thing every dead hypothesis")
            print("in RESEARCH.md §2 was measured against.")

    if args.oos:
        print("\n" + "=" * 84)
        print("OUT-OF-SAMPLE: does the ranking survive both halves?")
        print("=" * 84)
        orig = lab.load_pair

        def halved(which):
            def f(sym):
                rows = orig(sym)
                if not rows:
                    return rows
                mid = len(rows) // 2
                return rows[:mid] if which == "first" else rows[mid:]
            return f

        print(f"{'stop_atr':>9}{'horizon':>9}{'1st half':>12}{'2nd half':>12}")
        for w in WIDTHS:
            for h in HORIZONS:
                g = replace(shipped_geometry(), name="oos", stop_atr=w,
                            target_atr=args.rr * w, max_bars=h)
                lab.load_pair = halved("first")
                a, _, _ = breakeven_accuracy(g, h)
                lab.load_pair = halved("second")
                b, _, _ = breakeven_accuracy(g, h)
                lab.load_pair = orig
                fa = f"{a * 100:>11.1f}%" if np.isfinite(a) else f"{'n/a':>12}"
                fb = f"{b * 100:>11.1f}%" if np.isfinite(b) else f"{'n/a':>12}"
                print(f"{w:>9.1f}{h:>9}{fa}{fb}")


if __name__ == "__main__":
    main()
