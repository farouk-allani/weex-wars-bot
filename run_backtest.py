"""WEEX AI Wars II — Backtest Runner

Run: python run_backtest.py
"""

import sys
import yaml
import ccxt
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console

sys.path.insert(0, ".")

from src.core.models import Candle
from src.backtest.engine import Backtester, print_backtest_results

console = Console()


def fetch_historical_data(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    days: int = 90,
) -> list[Candle]:
    """Fetch historical candles from exchange."""
    console.print(f"[cyan]Fetching {days} days of {timeframe} data for {symbol}...[/]")

    exchange = ccxt.weex({"enableRateLimit": True})

    all_candles = []
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break

            all_candles.extend(ohlcv)
            since = ohlcv[-1][0] + 1

            if len(ohlcv) < 1000:
                break

        except Exception as e:
            console.print(f"[yellow]Error fetching data: {e}[/]")
            break

    candles = [
        Candle(
            timestamp=datetime.utcfromtimestamp(row[0] / 1000),
            open=row[1],
            high=row[2],
            low=row[3],
            close=row[4],
            volume=row[5],
        )
        for row in all_candles
    ]

    console.print(f"[green]Fetched {len(candles)} candles[/]")
    return candles


def main():
    """Run backtest on historical data."""
    # Load config
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Fetch data
    symbols = config.get("trading", {}).get("symbols", ["BTC/USDT:USDT"])
    symbol = symbols[0]  # Test on first symbol

    candles = fetch_historical_data(symbol, "1h", days=90)

    if len(candles) < 100:
        console.print("[red]Not enough data for backtest[/]")
        return

    # Run backtest
    console.print(f"\n[cyan]Running backtest on {symbol}...[/]")
    backtester = Backtester(config)
    result = backtester.run(candles, symbol)

    # Display results
    print_backtest_results(result)

    # Parameter optimization hint
    console.print("\n[bold yellow]💡 Optimization Tips:[/]")
    if result.win_rate < 0.50:
        console.print("  - Win rate < 50%: Increase RSI thresholds (try 80/20 instead of 75/25)")
    if result.max_drawdown > 0.15:
        console.print("  - Drawdown > 15%: Reduce max_risk_per_trade (try 1.5% instead of 2%)")
    if result.profit_factor < 1.5:
        console.print("  - Profit factor < 1.5: Tighten stop-losses (try 1.5x ATR instead of 2x)")
    if result.sharpe_ratio < 1.0:
        console.print("  - Sharpe < 1.0: Reduce trade frequency, increase minimum signal strength")


if __name__ == "__main__":
    main()
