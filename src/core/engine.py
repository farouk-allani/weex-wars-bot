"""WEEX AI Wars II — Main Trading Engine"""

import time
import yaml
import signal as sig
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

from .core.exchange import ExchangeClient
from .core.models import Side, Signal, Position, TradeResult
from .strategies.composite import CompositeStrategy
from .risk.manager import RiskManager
from .indicators.technical import calculate_atr

console = Console()


class TradingEngine:
    """Main trading engine — orchestrates everything."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.exchange = ExchangeClient(self.config)
        self.strategy = CompositeStrategy(self.config)
        self.risk = RiskManager(self.config)
        self.running = False
        self.cycle_count = 0

        # Graceful shutdown
        sig.signal(sig.SIGINT, self._shutdown)
        sig.signal(sig.SIGTERM, self._shutdown)

    def run(self):
        """Main trading loop."""
        self.running = True
        symbols = self.config.get("trading", {}).get("symbols", ["BTC/USDT:USDT"])
        timeframe = self.config.get("trading", {}).get("timeframe", "1h")
        lookback = self.config.get("trading", {}).get("lookback_periods", 100)

        console.print(Panel.fit(
            "[bold green]WEEX AI Wars II — Trading Bot[/]\n"
            f"Mode: [yellow]{self.config['trading']['mode']}[/]\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Timeframe: {timeframe}\n"
            f"Max Drawdown: {self.risk.max_drawdown:.0%}\n"
            f"Risk/Trade: {self.risk.max_risk_per_trade:.0%}",
            title="🤖 Bot Started",
        ))

        # Set leverage for all symbols
        leverage = self.config.get("trading", {}).get("default_leverage", 5)
        for symbol in symbols:
            self.exchange.set_leverage(symbol, leverage)

        while self.running:
            try:
                self.cycle_count += 1
                self._run_cycle(symbols, timeframe, lookback)
                self._display_status()

                # Sleep between cycles (1 minute for 1H timeframe)
                sleep_time = 60 if timeframe == "1h" else 30
                for _ in range(sleep_time):
                    if not self.running:
                        break
                    time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error in cycle: {e}[/]")
                time.sleep(30)

        self._cleanup()

    def _run_cycle(self, symbols: list[str], timeframe: str, lookback: int):
        """Single trading cycle."""
        account = self.exchange.get_account_state()

        # Check risk limits
        can_trade, reason = self.risk.can_trade(account)
        if not can_trade:
            console.print(f"[yellow]⚠ Trading blocked: {reason}[/]")
            return

        # Update existing positions
        self._manage_positions(account)

        # Analyze each symbol
        for symbol in symbols:
            try:
                # Skip if already have position in this symbol
                if any(p.symbol == symbol for p in account.positions):
                    continue

                # Fetch market data
                candles = self.exchange.fetch_candles(symbol, timeframe, lookback)
                if len(candles) < 50:
                    continue

                # Get funding rate
                funding_rate = self.exchange.fetch_funding_rate(symbol)

                # Generate signal
                signal = self.strategy.analyze(symbol, candles, funding_rate)
                if signal is None:
                    continue

                # Calculate position size
                size = self.risk.calculate_position_size(signal, account)
                if size <= 0:
                    continue

                # Execute trade
                self._execute_trade(signal, size, account)

            except Exception as e:
                console.print(f"[red]Error analyzing {symbol}: {e}[/]")

    def _execute_trade(self, signal: Signal, size: float, account):
        """Execute a trade based on signal."""
        console.print(f"\n[bold cyan]📊 Signal: {signal.side.value.upper()} {signal.symbol}[/]")
        console.print(f"   Strategy: {signal.strategy}")
        console.print(f"   Entry: ${signal.entry_price:.2f}")
        console.print(f"   Stop: ${signal.stop_loss:.2f}")
        console.print(f"   TP: ${signal.take_profit:.2f}")
        console.print(f"   R:R = {signal.risk_reward_ratio:.1f}")
        console.print(f"   Strength: {signal.strength:.2f}")
        console.print(f"   Size: {size:.4f}")
        console.print(f"   Reason: {signal.reason}")

        # Set leverage
        self.exchange.set_leverage(signal.symbol, signal.leverage)

        # Place order
        result = self.exchange.place_order(
            symbol=signal.symbol,
            side=signal.side,
            amount=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        if "error" in result:
            console.print(f"[red]   ❌ Order failed: {result['error']}[/]")
        else:
            console.print(f"[green]   ✅ Order filled: {result.get('id', 'N/A')}[/]")

    def _manage_positions(self, account):
        """Manage open positions (stop-loss, take-profit, trailing stops)."""
        candles_cache = {}

        for position in account.positions:
            try:
                # Fetch current price
                ticker = self.exchange.fetch_ticker(position.symbol)
                if not ticker:
                    continue

                current_price = ticker.get("last", position.entry_price)

                # Fetch candles for ATR calculation
                if position.symbol not in candles_cache:
                    candles_cache[position.symbol] = self.exchange.fetch_candles(
                        position.symbol, "1h", 20
                    )

                candles = candles_cache[position.symbol]
                if len(candles) >= 14:
                    highs = [c.high for c in candles]
                    lows = [c.low for c in candles]
                    closes = [c.close for c in candles]
                    atr = calculate_atr(
                        __import__("numpy").array(highs),
                        __import__("numpy").array(lows),
                        __import__("numpy").array(closes),
                    )[-1]
                else:
                    atr = current_price * 0.02  # Default 2% ATR

                # Update stops
                position = self.risk.adjust_stops(position, current_price, atr)

                # Check exit conditions
                if position.should_stop_loss(current_price):
                    console.print(f"[red]🔴 Stop-loss hit: {position.symbol}[/]")
                    self._close_position(position, current_price, "stop_loss")

                elif position.should_take_profit(current_price):
                    console.print(f"[green]🟢 Take-profit hit: {position.symbol}[/]")
                    self._close_position(position, current_price, "take_profit")

                elif position.should_trailing_stop(current_price):
                    console.print(f"[yellow]🟡 Trailing stop hit: {position.symbol}[/]")
                    self._close_position(position, current_price, "trailing_stop")

            except Exception as e:
                console.print(f"[red]Error managing {position.symbol}: {e}[/]")

    def _close_position(self, position: Position, current_price: float, reason: str):
        """Close a position and record the trade."""
        result = self.exchange.close_position(position.symbol)

        pnl = position.calculate_pnl(current_price)
        pnl_pct = (pnl / (position.size * position.entry_price / position.leverage)) * 100

        duration = int((datetime.utcnow() - position.opened_at).total_seconds())

        trade_result = TradeResult(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=current_price,
            size=position.size,
            leverage=position.leverage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_seconds=duration,
            exit_reason=reason,
        )

        self.risk.record_trade(trade_result)

        color = "green" if pnl >= 0 else "red"
        console.print(f"[{color}]   PnL: ${pnl:.2f} ({pnl_pct:.1f}%) — {reason}[/]")

    def _display_status(self):
        """Display current status dashboard."""
        if self.cycle_count % 5 != 0:  # Show every 5 cycles
            return

        account = self.exchange.get_account_state()
        stats = self.risk.get_stats()

        table = Table(title=f"📊 Status (Cycle #{self.cycle_count})", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Balance", f"${account.balance:.2f}")
        table.add_row("Equity", f"${account.equity:.2f}")
        table.add_row("Unrealized PnL", f"${account.unrealized_pnl:.2f}")
        table.add_row("Open Positions", str(len(account.positions)))
        table.add_row("Total Trades", str(stats.get("total_trades", 0)))
        table.add_row("Win Rate", f"{stats.get('win_rate', 0):.1%}")
        table.add_row("Total PnL", f"${stats.get('total_pnl', 0):.2f}")
        table.add_row("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.2f}")
        table.add_row("Max Drawdown", f"{stats.get('max_drawdown', 0):.1%}")
        table.add_row("Consecutive Losses", str(stats.get("consecutive_losses", 0)))

        console.print(table)

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            console.print(f"[yellow]Config not found at {path}, using defaults[/]")
            return {}

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _shutdown(self, signum, frame):
        """Graceful shutdown handler."""
        console.print("\n[yellow]Shutting down gracefully...[/]")
        self.running = False

    def _cleanup(self):
        """Cleanup on exit."""
        stats = self.risk.get_stats()
        console.print(Panel.fit(
            f"[bold]Final Statistics[/]\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1%}\n"
            f"Total PnL: ${stats.get('total_pnl', 0):.2f}\n"
            f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}",
            title="🏁 Session Ended",
        ))


def main():
    """Entry point."""
    console.print("[bold green]WEEX AI Wars II — Trading Bot[/]")
    console.print("[dim]Press Ctrl+C to stop[/]\n")

    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()
