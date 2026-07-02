"""WEEX AI Wars II — Main Trading Engine v2

Improvements:
1. Dynamic pair allocation by recent Sharpe
2. Passes existing positions to strategy for correlation guard
3. Better logging with pair performance tracking
4. Graceful degradation — if one pair errors, others continue
"""

import time
import yaml
import signal as sig
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .core.exchange import ExchangeClient
from .core.models import Side, Signal, Position, TradeResult
from .strategies.composite import CompositeStrategy
from .risk.manager import RiskManager
from .indicators.technical import calculate_atr
import numpy as np

console = Console()


class TradingEngine:
    """Main trading engine v2 — with dynamic allocation and correlation guard."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.exchange = ExchangeClient(self.config)
        self.strategy = CompositeStrategy(self.config)
        self.risk = RiskManager(self.config)
        self.running = False
        self.cycle_count = 0

        sig.signal(sig.SIGINT, self._shutdown)
        sig.signal(sig.SIGTERM, self._shutdown)

    def run(self):
        """Main trading loop."""
        self.running = True
        symbols = self.config.get("trading", {}).get("symbols", ["BTC/USDT:USDT"])
        timeframe = self.config.get("trading", {}).get("timeframe", "1h")
        lookback = self.config.get("trading", {}).get("lookback_periods", 100)

        console.print(Panel.fit(
            "[bold green]WEEX AI Wars II — Trading Bot v5[/]\n"
            f"Mode: [yellow]{self.config['trading']['mode']}[/]\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Timeframe: {timeframe}\n"
            f"Max Drawdown: {self.risk.max_drawdown:.0%}\n"
            f"Risk/Trade: {self.risk.max_risk_per_trade:.0%}\n"
            f"Features: Dynamic allocation, Correlation guard, Chandelier exit",
            title="🤖 Bot Started",
        ))

        leverage = self.config.get("trading", {}).get("default_leverage", 5)
        for symbol in symbols:
            self.exchange.set_leverage(symbol, leverage)

        while self.running:
            try:
                self.cycle_count += 1
                self._run_cycle(symbols, timeframe, lookback)
                self._display_status()

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
        """Single trading cycle with dynamic allocation."""
        account = self.exchange.get_account_state()

        can_trade, reason = self.risk.can_trade(account)
        if not can_trade:
            console.print(f"[yellow]⚠ Trading blocked: {reason}[/]")
            return

        self._manage_positions(account)

        # Build existing positions list for correlation guard
        existing = [(p.symbol, p.side.value) for p in account.positions]

        # Sort symbols by dynamic weight (trade best performers first)
        symbol_weights = {}
        for symbol in symbols:
            weight = self.risk.get_pair_weight(symbol)
            symbol_weights[symbol] = weight

        # Sort descending by weight
        sorted_symbols = sorted(symbols, key=lambda s: symbol_weights[s], reverse=True)

        for symbol in sorted_symbols:
            try:
                if any(p.symbol == symbol for p in account.positions):
                    continue

                candles = self.exchange.fetch_candles(symbol, timeframe, lookback)
                if len(candles) < 50:
                    continue

                funding_rate = self.exchange.fetch_funding_rate(symbol)

                # Pass existing positions for correlation check
                signal = self.strategy.analyze(symbol, candles, funding_rate, existing)
                if signal is None:
                    continue

                # Get dynamic pair weight
                pair_weight = self.risk.get_pair_weight(symbol)

                size = self.risk.calculate_position_size(signal, account, pair_weight)
                if size <= 0:
                    continue

                self._execute_trade(signal, size, account, pair_weight)

            except Exception as e:
                console.print(f"[red]Error analyzing {symbol}: {e}[/]")

    def _execute_trade(self, signal: Signal, size: float, account, pair_weight: float = 1.0):
        """Execute trade with logging."""
        console.print(f"\n[bold cyan]📊 Signal: {signal.side.value.upper()} {signal.symbol}[/]")
        console.print(f"   Strategy: {signal.strategy}")
        console.print(f"   Entry: ${signal.entry_price:.2f}")
        console.print(f"   Stop: ${signal.stop_loss:.2f}")
        console.print(f"   TP: ${signal.take_profit:.2f}")
        console.print(f"   R:R = {signal.risk_reward_ratio:.1f}")
        console.print(f"   Strength: {signal.strength:.2f}")
        console.print(f"   Size: {size:.4f}")
        console.print(f"   Pair weight: {pair_weight:.2f}x")
        console.print(f"   Reason: {signal.reason}")

        self.exchange.set_leverage(signal.symbol, signal.leverage)

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
        """Manage positions with chandelier exit."""
        candles_cache = {}

        for position in account.positions:
            try:
                ticker = self.exchange.fetch_ticker(position.symbol)
                if not ticker:
                    continue

                current_price = ticker.get("last", position.entry_price)

                if position.symbol not in candles_cache:
                    candles_cache[position.symbol] = self.exchange.fetch_candles(
                        position.symbol, "1h", 25
                    )

                candles = candles_cache[position.symbol]
                if len(candles) >= 14:
                    highs = np.array([c.high for c in candles])
                    lows = np.array([c.low for c in candles])
                    closes = np.array([c.close for c in candles])
                    atr = calculate_atr(highs, lows, closes)[-1]
                else:
                    atr = current_price * 0.02

                position = self.risk.adjust_stops(position, current_price, atr)

                if position.should_stop_loss(current_price):
                    console.print(f"[red]🔴 Stop-loss: {position.symbol}[/]")
                    self._close_position(position, current_price, "stop_loss")

                elif position.should_take_profit(current_price):
                    console.print(f"[green]🟢 Take-profit: {position.symbol}[/]")
                    self._close_position(position, current_price, "take_profit")

                elif position.should_trailing_stop(current_price):
                    console.print(f"[yellow]🟡 Trailing stop: {position.symbol}[/]")
                    self._close_position(position, current_price, "trailing_stop")

            except Exception as e:
                console.print(f"[red]Error managing {position.symbol}: {e}[/]")

    def _close_position(self, position: Position, current_price: float, reason: str):
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
        if self.cycle_count % 5 != 0:
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
        table.add_row("Sharpe", f"{stats.get('sharpe_ratio', 0):.2f}")
        table.add_row("Consec Losses", str(stats.get("consecutive_losses", 0)))
        table.add_row("Consec Wins", str(stats.get("consecutive_wins", 0)))

        # Per-pair performance
        pair_stats = stats.get("pair_stats", {})
        if pair_stats:
            table.add_row("", "")
            table.add_row("[bold]Pair Performance[/]", "")
            for symbol, ps in pair_stats.items():
                name = symbol.split("/")[0]
                table.add_row(
                    f"  {name}",
                    f"PnL=${ps['total_pnl']:.0f} | Sharpe={ps['sharpe']:.2f} | Weight={ps['weight']:.2f}x | Trades={ps['trades']}",
                )

        console.print(table)

    def _load_config(self, path: str) -> dict:
        config_path = Path(path)
        if not config_path.exists():
            return {}
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _shutdown(self, signum, frame):
        console.print("\n[yellow]Shutting down gracefully...[/]")
        self.running = False

    def _cleanup(self):
        stats = self.risk.get_stats()
        console.print(Panel.fit(
            f"[bold]Final Statistics[/]\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1%}\n"
            f"Total PnL: ${stats.get('total_pnl', 0):.2f}\n"
            f"Sharpe: {stats.get('sharpe_ratio', 0):.2f}",
            title="🏁 Session Ended",
        ))


def main():
    console.print("[bold green]WEEX AI Wars II — Trading Bot v5[/]")
    console.print("[dim]Press Ctrl+C to stop[/]\n")
    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()
