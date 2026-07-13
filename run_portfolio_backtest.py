"""Shared-capital multi-symbol portfolio backtest (v8.4).

Usage:
  python run_portfolio_backtest.py
  python run_portfolio_backtest.py --days 90
"""

import argparse
import sys
import yaml
import ccxt
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, ".")

from src.core.models import Candle
from src.backtest.engine import print_backtest_results, resample_to_htf
from src.backtest.portfolio import PortfolioBacktester

console = Console()


def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> list[Candle]:
    console.print(f"[cyan]Fetching {days}d {timeframe} {symbol}...[/]")
    exchange = ccxt.binance({"enableRateLimit": True})
    all_rows = []
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_rows.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            console.print(f"[yellow]Fetch error: {e}[/]")
            break
    candles = [
        Candle(
            timestamp=datetime.utcfromtimestamp(r[0] / 1000),
            open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
        )
        for r in all_rows
    ]
    console.print(f"[green]  → {len(candles)} bars[/]")
    return candles


def fetch_funding(symbol: str, days: int) -> dict:
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        data = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        return {
            fr["timestamp"]: fr.get("fundingRate", 0.0)
            for fr in (data or [])
            if fr.get("timestamp")
        }
    except Exception:
        return {}


def interpolate_funding(timestamps, rates: dict) -> list[float]:
    if not rates:
        return [0.0] * len(timestamps)
    sorted_ts = sorted(rates.keys())
    out, last = [], 0.0
    for ts in timestamps:
        ms = int(ts.timestamp() * 1000)
        for t in reversed(sorted_ts):
            if t <= ms:
                last = rates[t]
                break
        out.append(last)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    symbols = config.get("trading", {}).get("symbols", [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"
    ])

    market = {}
    for sym in symbols:
        c1h = fetch_ohlcv(sym, "1h", args.days)
        if len(c1h) < 150:
            console.print(f"[red]Skip {sym}[/]")
            continue
        fr = fetch_funding(sym, args.days)
        funding = interpolate_funding([c.timestamp for c in c1h], fr)
        market[sym] = {
            "candles": c1h,
            "funding": funding,
            "htf": resample_to_htf(c1h, 4),
        }

    if not market:
        console.print("[red]No data[/]")
        return

    console.print(Panel.fit(
        f"[bold]Portfolio backtest[/]\n"
        f"Symbols: {', '.join(market.keys())}\n"
        f"Days: {args.days} | Shared capital $10k | max positions enforced",
        title="v8.4",
    ))

    bt = PortfolioBacktester(config)
    result = bt.run(market)
    print_backtest_results(result)

    # Per-symbol from trades
    by_sym = {}
    for t in result.trades:
        by_sym.setdefault(t.symbol, []).append(t.pnl)
    if by_sym:
        console.print("\n[bold cyan]Per-symbol PnL (shared book)[/]")
        for sym, pnls in by_sym.items():
            console.print(
                f"  {sym.split('/')[0]}: ${sum(pnls):+.2f}  n={len(pnls)}  "
                f"WR={sum(1 for p in pnls if p>0)/len(pnls):.0%}"
            )


if __name__ == "__main__":
    main()
