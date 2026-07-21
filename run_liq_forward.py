"""Forward cascade analysis on REAL Binance forced orders.

This is the ground-truth test the OI-proxy scan (run_liq_scan.py) could not
do: it confused voluntary de-risking with forced liquidation. Here every
event is a confirmed forced order from Binance's forceOrder stream, collected
live by run_liq_record.py.

Binance side convention on forceOrder: a liquidation with side "sell" is a
LONG being force-closed (sells into the book -> price DUMPS); side "buy" is a
SHORT being squeezed (buys -> price PUMPS). We FADE: buy the long-flush,
sell the short-squeeze.

Event (liquidation-native, pre-declared by mechanism — NOT fitted):
  in a rolling W-second window, net one-sided forced-liquidation notional
  >= T dollars AND the dominant side is >= 70% of window notional.
  Direction = fade the dominant side. Cooldown = longest horizon.
Forward return measured on the SAME symbol's mid tape (Binance mids, 1s,
throttled) at 5/15/30/60 min, in the fade direction.

We test a small T grid to read the DOSE-RESPONSE (the robust signal in the
proxy scan: bigger flush -> bigger bounce) and split every cell in half for
OOS. Judge on monotonicity + both-halves + net-of-cost, not one green cell.
Costs: WEEX maker 0.04% RT, taker 0.12% RT.

    python run_liq_forward.py data/liq_forward/*.jsonl
"""

import glob
import json
import sys
from collections import defaultdict

import numpy as np

MAKER_RT = 0.0004
TAKER_RT = 0.0012
GRID_S = 5
HORIZONS = {"5m": 300, "15m": 900, "30m": 1800, "60m": 3600}
WINDOW_S = 180                       # rolling cascade window
THRESHOLDS = [1e5, 2.5e5, 5e5, 1e6]  # net one-sided notional, dose-response grid
DOMINANCE = 0.70
PRICE_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "LTC/USDT:USDT",
]


def load(paths):
    liqs = defaultdict(list)   # sym -> [(rt_ms, side, usd)]
    px = defaultdict(list)     # sym -> [(rt_ms, mid)]
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if '"ch"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ch = r.get("ch")
                if ch == "liq" and r.get("usd"):
                    liqs[r["sym"]].append((r["rt"], r["side"], r["usd"]))
                elif ch == "px" and r.get("mid"):
                    px[r["sym"]].append((r["rt"], r["mid"]))
    for d in (liqs, px):
        for s in d:
            d[s].sort(key=lambda x: x[0])
    return liqs, px


def grid_mid(rows):
    """Forward-filled mid on a GRID_S-second grid. Returns (t0_ms, arr)."""
    if len(rows) < 100:
        return None, None
    t0, t1 = rows[0][0], rows[-1][0]
    n = int((t1 - t0) // (GRID_S * 1000)) + 1
    arr = np.full(n, np.nan)
    for t, m in rows:
        arr[int((t - t0) // (GRID_S * 1000))] = m
    idx = np.where(~np.isnan(arr))[0]
    last = arr[idx[0]]
    for i in range(idx[0], n):
        if np.isnan(arr[i]):
            arr[i] = last
        else:
            last = arr[i]
    arr[: idx[0]] = arr[idx[0]]
    return t0, arr


def find_cascades(liq_rows, thresh):
    """Rolling WINDOW_S net one-sided notional >= thresh, dominant >=70%.
    Returns [(t_ms, fade_sign)] where fade_sign +1 = buy (long flush)."""
    events = []
    buf = []  # (t, signed_usd) sell=-, buy=+
    max_h = max(HORIZONS.values()) * 1000
    last_ev = -10**18
    for t, side, usd in liq_rows:
        s = -usd if side == "sell" else usd  # sell-liq pushes price down
        buf.append((t, s))
        cutoff = t - WINDOW_S * 1000
        while buf and buf[0][0] < cutoff:
            buf.pop(0)
        pos = sum(x for _, x in buf if x > 0)
        neg = -sum(x for _, x in buf if x < 0)
        net = pos - neg
        total = pos + neg
        if total <= 0:
            continue
        dom = max(pos, neg) / total
        if abs(net) >= thresh and dom >= DOMINANCE and t - last_ev > max_h:
            # net>0 means squeeze (price up) -> fade SELL (-1);
            # net<0 means long flush (price down) -> fade BUY (+1)
            events.append((t, -np.sign(net)))
            last_ev = t
    return events


def raw_ret(grids, sym, t_ms, sec):
    t0, mid = grids[sym]
    i = int((t_ms - t0) // (GRID_S * 1000))
    j = i + sec // GRID_S
    if i < 0 or j >= len(mid):
        return None
    return np.log(mid[j] / mid[i])


def market_ret(grids, t_ms, sec):
    """Equal-weight raw return across all pairs over [t, t+sec] — the beta."""
    rs = [raw_ret(grids, s, t_ms, sec) for s in grids]
    rs = [r for r in rs if r is not None]
    return float(np.mean(rs)) if rs else None


def fwd_returns(events, sym, grids, beta_neutral=False):
    """Faded forward returns. beta_neutral subtracts fade*market_ret so a
    result can't be just the whole market drifting during a downtrend."""
    out = {h: [] for h in HORIZONS}
    for t_ms, fade in events:
        for h, sec in HORIZONS.items():
            r = raw_ret(grids, sym, t_ms, sec)
            if r is None:
                continue
            if beta_neutral:
                mkt = market_ret(grids, t_ms, sec)
                if mkt is None:
                    continue
                out[h].append(fade * (r - mkt))
            else:
                out[h].append(fade * r)
    return out


def dedupe_cross(events_by_sym, window_ms=300_000):
    """Collapse cross-symbol events within window into one market episode
    (keep all, but tag so pooled t isn't inflated by correlated names).
    Returns a flat list of (t_ms, sym, fade) with correlated cluster thinned
    to the single largest-|net| — approximated here by keeping the earliest
    per cluster across symbols."""
    flat = sorted(
        ((t, s, f) for s, evs in events_by_sym.items() for (t, f) in evs),
        key=lambda x: x[0],
    )
    kept, last_t = [], -10**18
    for t, s, f in flat:
        if t - last_t >= window_ms:
            kept.append((t, s, f))
            last_t = t
    return kept


def summarize(all_rets, label, cost):
    if not all_rets or not all_rets[next(iter(HORIZONS))]:
        print(f"  {label}: no events")
        return
    print(f"\n  {label}")
    print(f"  {'horizon':>8} {'n':>5} {'mean':>9} {'t':>6} {'1st half':>9} "
          f"{'2nd half':>9} {'win%':>5} {'net maker':>10}")
    for h in HORIZONS:
        r = np.array(all_rets[h])
        if len(r) < 4:
            print(f"  {h:>8} {len(r):>5}  (too few)")
            continue
        half = len(r) // 2
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if r.std() > 0 else 0
        nm = r.mean() - cost
        flag = "  <==" if (nm > 0 and abs(t) >= 2) else ""
        print(f"  {h:>8} {len(r):>5} {r.mean()*100:>+8.3f}% {t:>6.1f} "
              f"{r[:half].mean()*100:>+8.3f}% {r[half:].mean()*100:>+8.3f}% "
              f"{(r>0).mean()*100:>4.0f}% {nm*100:>+9.3f}%{flag}")


def main():
    import argparse
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--since", default=None,
                    help="ISO date: only use data at/after this UTC date. "
                         "REQUIRED discipline for the pre-declared continuation "
                         "hypothesis: judge it only on data after 2026-07-20.")
    ap.add_argument("--direction", choices=["fade", "with"], default="fade",
                    help="'fade' = against the flush (KILLED 2026-07-20); "
                         "'with' = continuation, the registered follow-up")
    args = ap.parse_args()

    paths = []
    for a in args.globs:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        print("usage: run_liq_forward.py data/liq_forward/*.jsonl "
              "[--since 2026-07-20 --direction with]")
        sys.exit(1)
    print(f"loading {len(paths)} files...")
    liqs, px = load(paths)

    if args.since:
        cut = int(datetime.fromisoformat(args.since)
                  .replace(tzinfo=timezone.utc).timestamp() * 1000)
        liqs = {s: [r for r in v if r[0] >= cut] for s, v in liqs.items()}
        liqs = {s: v for s, v in liqs.items() if v}
        px = {s: [r for r in v if r[0] >= cut] for s, v in px.items()}
        px = {s: v for s, v in px.items() if len(v) > 100}
        print(f"--since {args.since}: filtered to fresh data only")
    sign = -1.0 if args.direction == "with" else 1.0
    if args.direction == "with":
        print("DIRECTION = WITH the cascade (continuation hypothesis, "
              "pre-declared 2026-07-20: primary cell $250k/60m beta-neutral episodes)")
    total_liq = sum(len(v) for v in liqs.values())
    total_usd = sum(u for v in liqs.values() for _, _, u in v)
    span_h = (max(v[-1][0] for v in liqs.values() if v) -
              min(v[0][0] for v in liqs.values() if v)) / 3600000
    print(f"{total_liq:,} forced orders, ${total_usd:,.0f} notional, "
          f"{span_h:.1f}h span, {len(liqs)} symbols with liqs, "
          f"{sum(1 for s in PRICE_SYMBOLS if s in px)} of 8 pairs have mid tape")

    grids = {}
    for s in PRICE_SYMBOLS:
        if s in px:
            t0, arr = grid_mid(px[s])
            if arr is not None:
                grids[s] = (t0, arr)

    print("\n" + "=" * 78)
    print("DOSE-RESPONSE: fade real liquidation cascades on the 8 competition pairs")
    print("(fade = buy long-flush / sell short-squeeze)")
    print("=" * 78)
    for thr in THRESHOLDS:
        events_by_sym = {}
        for s in PRICE_SYMBOLS:
            if s in grids and s in liqs:
                ev = [(t, f * sign) for t, f in find_cascades(liqs[s], thr)]
                if ev:
                    events_by_sym[s] = ev
        n_ev = sum(len(v) for v in events_by_sym.values())

        pooled_raw = {h: [] for h in HORIZONS}
        pooled_bn = {h: [] for h in HORIZONS}
        for s, ev in events_by_sym.items():
            fr = fwd_returns(ev, s, grids, beta_neutral=False)
            fb = fwd_returns(ev, s, grids, beta_neutral=True)
            for h in HORIZONS:
                pooled_raw[h].extend(fr[h])
                pooled_bn[h].extend(fb[h])

        episodes = dedupe_cross(events_by_sym)
        ep_bn = {h: [] for h in HORIZONS}
        for t, s, f in episodes:
            fb = fwd_returns([(t, f)], s, grids, beta_neutral=True)
            for h in HORIZONS:
                ep_bn[h].extend(fb[h])

        print(f"\n--- threshold ${thr/1e3:.0f}k net / {WINDOW_S}s: "
              f"{n_ev} events, {len(episodes)} independent episodes ---")
        summarize(pooled_raw, "RAW (drift-contaminated upper bound)", MAKER_RT)
        summarize(pooled_bn, "BETA-NEUTRAL (minus equal-weight market move)", MAKER_RT)
        summarize(ep_bn, "EPISODES beta-neutral (deduped -> honest t)", MAKER_RT)

    print("\n[bar] a real edge = monotonic dose-response (bigger flush -> bigger bounce) "
          "AND net-positive at maker AND same sign both halves AND |t|>=2. Forward returns "
          "on Binance mid (reversion proxy); WEEX execution slippage is the separate live gate.")


if __name__ == "__main__":
    main()
