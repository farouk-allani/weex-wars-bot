#!/usr/bin/env python3
"""
Robustness gate for the exit-geometry result.

The break-even numbers only matter if they survive the checks that killed every
previous "edge" in this repo: out-of-sample split, per-pair breakdown, and
parameter sensitivity. A geometry that only wins on the full sample with one
stop/target pair is a curve fit, not a fix.

Also tests protection-preserving middles: `pure_bracket` wins on paper but gives
back open profit, which the competition's stability metric punishes. If a variant
that keeps a breakeven stop and a trail gets close to pure_bracket's bar, it is the
better real-world choice.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import run_exit_lab as lab
from run_exit_lab import make_oracle, run

CURRENT = lab.CURRENT

VARIANTS = {
    "current": CURRENT,
    # drop the partial only
    "no_partial": replace(CURRENT, name="no_partial", partial_enabled=False),
    # drop partial, and stop moving BE until the trade has real room (2R)
    "no_partial_be2R": replace(CURRENT, name="no_partial_be2R", partial_enabled=False,
                               be_trigger_r=2.0, trail_takes_tighter=False),
    # keep protection, but let the ATR chandelier bind instead of the 0.6% percent trail
    "no_partial_chand": replace(CURRENT, name="no_partial_chand", partial_enabled=False,
                                be_trigger_r=2.0, trail_act=0.008,
                                trail_pct=0.99, trail_pct_post=0.99,
                                trail_takes_tighter=False),
    "pure_bracket": replace(CURRENT, name="pure_bracket", partial_enabled=False,
                            be_trigger_r=99, trail_act=99),
}

SEEDS = [7, 11, 23, 41]
PROBE = 0.56  # a realistic "we found something" accuracy


def mean_net(g, entry_key: str, seeds=SEEDS) -> float:
    return float(np.mean([run(g, entry_key, seed=s)["net_per_trade"] for s in seeds]))


def section(title: str):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main():
    lab.MAKER_EXITS = True
    lab.ENTRY_FNS["_p"] = make_oracle(PROBE)

    orig_load = lab.load_pair

    # ---------------------------------------------------------- OOS split
    section(f"1. OUT-OF-SAMPLE  (net $/trade @ {PROBE:.0%} accuracy, maker TP legs)")

    def halved(which: str):
        def f(sym):
            rows = orig_load(sym)
            if not rows:
                return rows
            mid = len(rows) // 2
            return rows[:mid] if which == "first" else rows[mid:]
        return f

    print(f"{'variant':<20}{'1st half':>12}{'2nd half':>12}{'full':>12}   consistent?")
    for name, g in VARIANTS.items():
        lab.load_pair = halved("first")
        a = mean_net(g, "_p")
        lab.load_pair = halved("second")
        b = mean_net(g, "_p")
        lab.load_pair = orig_load
        c = mean_net(g, "_p")
        ok = "yes" if (a > 0) == (b > 0) else "NO - sign flips"
        print(f"{name:<20}{a:>+12.4f}{b:>+12.4f}{c:>+12.4f}   {ok}")

    # ---------------------------------------------------------- per pair
    section(f"2. PER-PAIR  (current vs pure_bracket, net $/trade @ {PROBE:.0%})")
    all_pairs = list(lab.PAIRS)
    print(f"{'pair':<8}{'current':>12}{'no_partial':>14}{'pure_bracket':>15}   winner")
    wins = {"current": 0, "no_partial": 0, "pure_bracket": 0}
    for p in all_pairs:
        lab.PAIRS = [p]
        vals = {n: mean_net(VARIANTS[n], "_p") for n in wins}
        lab.PAIRS = all_pairs
        best = max(vals, key=vals.get)
        wins[best] += 1
        print(f"{p:<8}{vals['current']:>+12.4f}{vals['no_partial']:>+14.4f}"
              f"{vals['pure_bracket']:>+15.4f}   {best}")
    print(f"\npair-level wins: {wins}")

    # ------------------------------------------------- parameter sensitivity
    section("3. STOP/TARGET SENSITIVITY  (is pure_bracket a lucky corner?)")
    print(f"{'stop_atr':>9}{'target_atr':>12}{'current':>12}{'pure_bracket':>15}   delta")
    beats = 0
    total = 0
    for s_atr in (0.8, 1.2, 1.6, 2.0):
        for t_atr in (1.6, 2.4, 3.2):
            a = mean_net(replace(VARIANTS["current"], stop_atr=s_atr, target_atr=t_atr),
                         "_p", seeds=SEEDS[:2])
            b = mean_net(replace(VARIANTS["pure_bracket"], stop_atr=s_atr, target_atr=t_atr),
                         "_p", seeds=SEEDS[:2])
            total += 1
            if b > a:
                beats += 1
            print(f"{s_atr:>9.1f}{t_atr:>12.1f}{a:>+12.4f}{b:>+15.4f}   {b - a:>+.4f}")
    print(f"\npure_bracket beats current in {beats}/{total} stop/target combinations")

    # ---------------------------------------------------------- zero-alpha floor
    section("4. ZERO-ALPHA CONTROL  (must be ~ -costs for all; large gap = bug)")
    for name, g in VARIANTS.items():
        r = run(g, "random", seed=7)
        print(f"{name:<20} net={r['net_per_trade']:+.4f}  gross={r['gross_per_trade']:+.4f}"
              f"  WR={r['winrate']:.1f}%  payoff={r['payoff']:.2f}"
              f"  sharpe={r['sharpe_per_trade']:+.3f}")


if __name__ == "__main__":
    main()
