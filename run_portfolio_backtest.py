"""Shared-capital multi-symbol portfolio backtest (v8.4).

Usage:
  python run_portfolio_backtest.py
  python run_portfolio_backtest.py --days 90
"""

import argparse
import sys
import yaml
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, ".")

from src.backtest.engine import print_backtest_results, resample_to_htf
from src.backtest.portfolio import PortfolioBacktester
from src.data.fetcher import fetch_ohlcv, fetch_funding_map, interpolate_funding

console = Console()


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
        console.print(f"[cyan]Loading {sym} ({args.days}d)...[/]")
        c1h = fetch_ohlcv(sym, "1h", args.days, use_cache=True)
        if len(c1h) < 150:
            console.print(f"[red]Skip {sym}[/]")
            continue
        fr = fetch_funding_map(sym, args.days, use_cache=True)
        funding = interpolate_funding([c.timestamp for c in c1h], fr)
        market[sym] = {
            "candles": c1h,
            "funding": funding,
            "htf": resample_to_htf(c1h, 4),
        }
        console.print(f"[green]  → {len(c1h)} bars[/]")

    if not market:
        console.print("[red]No data[/]")
        return

    console.print(Panel.fit(
        f"[bold]Portfolio backtest[/]\n"
        f"Symbols: {', '.join(market.keys())}\n"
        f"Days: {args.days} | Shared capital $10k | max positions enforced",
        title="v8.5",
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
