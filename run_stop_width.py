#!/usr/bin/env python3
"""
Is the stop too tight? — the lever that outranked exit geometry.

Under risk-based sizing (constant dollar risk at the stop, as the live sizer does),
a WIDER stop means a SMALLER position, which means proportionally lower fees AND a
lower chance of being noise-stopped before a real move resolves. That is a free
improvement if it holds up; it costs nothing and needs no alpha.

Gates applied, because every previous "edge" in this repo died at one of them:
  * zero-alpha control - at 50% accuracy the only thing separating stop widths
    should be cost, so the curve must be shallow and monotonic, not a cliff
  * out-of-sample halves - the ranking must survive both
  * per-pair - must not be one pair carrying it
  * position-cap check - max_position_pct truncates size at tight stops, which
    quietly reduces their real risk and FLATTERS them; reported so it is visible
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import run_exit_lab as lab
from run_exit_lab import make_oracle, run

WIDTHS = [0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0]
SEEDS = [7, 11, 23, 41]
GEOMS = {
    "current": lab.CURRENT,
    "pure_bracket": replace(lab.CURRENT, name="pure_bracket", partial_enabled=False,
                            be_trigger_r=99, trail_act=99),
}


def mean_net(g, key: str, seeds=SEEDS) -> float:
    return float(np.mean([run(g, key, seed=s)["net_per_trade"] for s in seeds]))


def notional_at(stop_atr: float) -> float:
    """What the live sizer would actually put on, given a typical 1% ATR."""
    atr_frac = 0.01
    risk_frac = stop_atr * atr_frac
    notional = lab.EQUITY * lab.MAX_RISK_PER_TRADE / risk_frac
    cap = lab.EQUITY * lab.MAX_POSITION_PCT
    return min(notional, cap)


def main():
    lab.MAKER_EXITS = True
    lab.ENTRY_FNS["_p"] = make_oracle(0.56)
    lab.ENTRY_FNS["_flat"] = make_oracle(0.50)
    orig_load = lab.load_pair
    all_pairs = list(lab.PAIRS)

    print("=" * 78)
    print("STOP WIDTH  (risk-sized: wider stop -> smaller position, same $ risk)")
    print("=" * 78)
    print(f"{'stop_atr':>9}{'notional$':>11}{'capped?':>9}"
          f"{'@50% (cost only)':>19}{'@56% current':>15}{'@56% pure':>12}")
    for w in WIDTHS:
        n = notional_at(w)
        capped = "YES" if n >= lab.EQUITY * lab.MAX_POSITION_PCT - 1e-9 else ""
        g_cur = replace(GEOMS["current"], stop_atr=w)
        g_pure = replace(GEOMS["pure_bracket"], stop_atr=w)
        flat = mean_net(g_cur, "_flat", seeds=SEEDS[:2])
        a = mean_net(g_cur, "_p")
        b = mean_net(g_pure, "_p")
        print(f"{w:>9.1f}{n:>11.0f}{capped:>9}{flat:>19.4f}{a:>15.4f}{b:>12.4f}")

    print("\nNote: 'capped?' means max_position_pct truncated the size, so that row "
          "carries\nLESS than the nominal risk - it flatters the tight stops, "
          "not the wide ones.")

    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE  (pure_bracket, net $/trade @ 56%)")
    print("=" * 78)

    def halved(which):
        def f(sym):
            rows = orig_load(sym)
            if not rows:
                return rows
            mid = len(rows) // 2
            return rows[:mid] if which == "first" else rows[mid:]
        return f

    print(f"{'stop_atr':>9}{'1st half':>12}{'2nd half':>12}{'full':>12}")
    for w in WIDTHS:
        g = replace(GEOMS["pure_bracket"], stop_atr=w)
        lab.load_pair = halved("first")
        a = mean_net(g, "_p", seeds=SEEDS[:2])
        lab.load_pair = halved("second")
        b = mean_net(g, "_p", seeds=SEEDS[:2])
        lab.load_pair = orig_load
        c = mean_net(g, "_p", seeds=SEEDS[:2])
        print(f"{w:>9.1f}{a:>12.4f}{b:>12.4f}{c:>12.4f}")

    print("\n" + "=" * 78)
    print("PER-PAIR  (pure_bracket @ 56%: does every pair prefer a wider stop?)")
    print("=" * 78)
    print(f"{'pair':<7}" + "".join(f"{w:>9.1f}" for w in WIDTHS) + "   best")
    prefer_wide = 0
    for p in all_pairs:
        lab.PAIRS = [p]
        vals = [mean_net(replace(GEOMS["pure_bracket"], stop_atr=w), "_p", seeds=SEEDS[:2])
                for w in WIDTHS]
        lab.PAIRS = all_pairs
        best = WIDTHS[int(np.argmax(vals))]
        if best >= 1.6:
            prefer_wide += 1
        print(f"{p:<7}" + "".join(f"{v:>9.3f}" for v in vals) + f"   {best}")
    print(f"\n{prefer_wide}/{len(all_pairs)} pairs prefer a stop of 1.6 ATR or wider "
          f"(live config uses ~1.2)")


if __name__ == "__main__":
    main()
