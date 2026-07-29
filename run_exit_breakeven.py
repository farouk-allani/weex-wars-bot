#!/usr/bin/env python3
"""
How much directional edge does each exit geometry need to break even?

run_exit_lab.py showed the geometries rank consistently; this quantifies the thing
that actually decides whether the bot can ever be profitable: the minimum entry
accuracy each geometry demands. A geometry that needs 62% is unreachable. One that
needs 54% is a research target.

Also prices the cost lever (--maker-exits), because on zero-alpha entries the entire
loss was costs, and costs are the one input fully under our control.
"""

from __future__ import annotations

import numpy as np

import run_exit_lab as lab
from run_exit_lab import Geometry, make_oracle, run
from dataclasses import replace

CURRENT = lab.CURRENT
CANDIDATES = {
    "current": CURRENT,
    "no_partial": replace(CURRENT, name="no_partial", partial_enabled=False),
    "pure_bracket": replace(CURRENT, name="pure_bracket", partial_enabled=False,
                            be_trigger_r=99, trail_act=99),
}

EDGES = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
SEEDS = [7, 11, 23]


def sweep(maker: bool) -> dict:
    lab.MAKER_EXITS = maker
    out: dict[str, list[float]] = {k: [] for k in CANDIDATES}
    for p in EDGES:
        lab.ENTRY_FNS["_probe"] = make_oracle(p)
        for name, g in CANDIDATES.items():
            vals = [run(g, "_probe", seed=s)["net_per_trade"] for s in SEEDS]
            out[name].append(float(np.mean(vals)))
    return out


def breakeven(edges: list[float], nets: list[float]) -> float | None:
    """Linear interpolation of the accuracy at which net/trade crosses zero."""
    for i in range(len(nets) - 1):
        if nets[i] <= 0 <= nets[i + 1]:
            f = -nets[i] / (nets[i + 1] - nets[i])
            return edges[i] + f * (edges[i + 1] - edges[i])
    if nets[-1] > 0:
        return edges[-1]
    # extrapolate off the last segment
    slope = (nets[-1] - nets[-2]) / (edges[-1] - edges[-2])
    return edges[-1] + (-nets[-1] / slope) if slope > 0 else None


def main():
    for maker in (False, True):
        tag = "maker TP/partial legs (stops stay taker)" if maker else "all-taker exits (LIVE TODAY)"
        print(f"\n=== net $/trade on $300 notional | {tag} ===")
        print("accuracy   " + "".join(f"{n:>15}" for n in CANDIDATES))
        res = sweep(maker)
        for i, p in enumerate(EDGES):
            print(f"  {p:.0%}     " + "".join(f"{res[n][i]:>+15.4f}" for n in CANDIDATES))
        print("break-even " + "".join(
            (f"{breakeven(EDGES, res[n]):>14.1%} " if breakeven(EDGES, res[n]) else f"{'>60%':>15}")
            for n in CANDIDATES))


if __name__ == "__main__":
    main()
