"""WEEX AI Wars II — Main Trading Engine v8

- Multi-timeframe data (1h signal + 4h confirmation)
- SL/TP always passed into exchange
- Strategy win/loss recording
- Local bracket sync for live trailing stops
- Graceful per-symbol error isolation
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

from .exchange import ExchangeClient
from .models import Side, Signal, Position, TradeResult
from ..strategies.composite import CompositeStrategy
from ..risk.manager import RiskManager
from ..indicators.technical import calculate_atr
import numpy as np

console = Console()


class TradingEngine:
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
        self.running = True
        symbols = self.config.get("trading", {}).get("symbols", ["BTC/USDT:USDT"])
        timeframe = self.config.get("trading", {}).get("timeframe", "1h")
        htf = self.config.get("trading", {}).get("higher_timeframe", "4h")
        lookback = self.config.get("trading", {}).get("lookback_periods", 100)
        htf_lookback = self.config.get("trading", {}).get("htf_lookback", 80)

        console.print(Panel.fit(
            "[bold green]WEEX AI Wars II — Trading Bot v8[/]\n"
            f"Mode: [yellow]{self.config['trading']['mode']}[/]\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Timeframes: {timeframe} + {htf}\n"
            f"Max Drawdown: {self.risk.max_drawdown:.0%}\n"
            f"Risk/Trade: {self.risk.max_risk_per_trade:.1%}\n"
            f"Features: HTF confluence, strength sizing, fixed SL/TP, smart keep-alive",
            title="Bot Started",
        ))

        leverage = self.config.get("trading", {}).get("default_leverage", 5)
        for symbol in symbols:
            self.exchange.set_leverage(symbol, leverage)

        while self.running:
            try:
                self.cycle_count += 1
                self._run_cycle(symbols, timeframe, htf, lookback, htf_lookback)
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

    def _run_cycle(self, symbols, timeframe, htf, lookback, htf_lookback):
        account = self.exchange.get_account_state()

        can_trade, reason = self.risk.can_trade(account)
        if not can_trade:
            console.print(f"[yellow]Trading blocked: {reason}[/]")
            # Still manage open positions even when new entries blocked
            self._manage_positions(account)
            return

        self._manage_positions(account)
        account = self.exchange.get_account_state()

        existing = [(p.symbol, p.side.value) for p in account.positions]
        symbol_weights = {s: self.risk.get_pair_weight(s) for s in symbols}
        sorted_symbols = sorted(symbols, key=lambda s: symbol_weights[s], reverse=True)

        for symbol in sorted_symbols:
            try:
                if any(p.symbol == symbol for p in account.positions):
                    continue
                if len(account.positions) >= self.risk.max_open_positions:
                    break

                candles = self.exchange.fetch_candles(symbol, timeframe, lookback)
                if len(candles) < 100:
                    continue

                htf_candles = self.exchange.fetch_candles(symbol, htf, htf_lookback)
                funding_rate = self.exchange.fetch_funding_rate(symbol)

                signal = self.strategy.analyze(
                    symbol, candles, funding_rate, existing,
                    higher_tf_candles=htf_candles if htf_candles else None,
                )
                if signal is None:
                    continue

                pair_weight = self.risk.get_pair_weight(symbol)
                size = self.risk.calculate_position_size(signal, account, pair_weight)
                if size <= 0:
                    continue

                self._execute_trade(signal, size, account, pair_weight)
                # Refresh account after fill
                account = self.exchange.get_account_state()
                existing = [(p.symbol, p.side.value) for p in account.positions]

            except Exception as e:
                console.print(f"[red]Error analyzing {symbol}: {e}[/]")

    def _execute_trade(self, signal: Signal, size: float, account, pair_weight: float = 1.0):
        console.print(f"\n[bold cyan]Signal: {signal.side.value.upper()} {signal.symbol}[/]")
        console.print(f"   Strategy: {signal.strategy}")
        console.print(f"   Entry: ${signal.entry_price:.4f}")
        console.print(f"   Stop: ${signal.stop_loss:.4f}")
        console.print(f"   TP: ${signal.take_profit:.4f}")
        console.print(f"   R:R = {signal.risk_reward_ratio:.2f}")
        console.print(f"   Strength: {signal.strength:.2f}")
        console.print(f"   Size: {size:.6f}")
        console.print(f"   Pair weight: {pair_weight:.2f}x")
        console.print(f"   Reason: {signal.reason}")

        self.exchange.set_leverage(signal.symbol, signal.leverage)

        result = self.exchange.place_order(
            symbol=signal.symbol,
            side=signal.side,
            amount=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy,
            leverage=signal.leverage,
        )

        if "error" in result:
            console.print(f"[red]   Order failed: {result['error']}[/]")
            return

        console.print(f"[green]   Order filled: {result.get('id', 'N/A')}[/]")
        if result.get("sl_placed") is False:
            console.print("[yellow]   SL not on exchange — software stop active[/]")
        if result.get("tp_placed") is False:
            console.print("[yellow]   TP not on exchange — software TP active[/]")

    def _manage_positions(self, account):
        candles_cache = {}

        for position in list(account.positions):
            try:
                ticker = self.exchange.fetch_ticker(position.symbol)
                if not ticker:
                    continue

                current_price = float(ticker.get("last") or position.entry_price)

                if position.symbol not in candles_cache:
                    candles_cache[position.symbol] = self.exchange.fetch_candles(
                        position.symbol, "1h", 30
                    )

                candles = candles_cache[position.symbol]
                if len(candles) >= 14:
                    highs = np.array([c.high for c in candles])
                    lows = np.array([c.low for c in candles])
                    closes = np.array([c.close for c in candles])
                    atr = float(calculate_atr(highs, lows, closes)[-1])
                else:
                    atr = current_price * 0.015

                position = self.risk.adjust_stops(position, current_price, atr)
                self.exchange.update_local_brackets(position)

                # Paper: keep in-memory position synced
                if self.exchange.mode == "paper" and position.symbol in self.exchange.paper_positions:
                    self.exchange.paper_positions[position.symbol] = position

                if position.should_stop_loss(current_price):
                    console.print(f"[red]Stop-loss: {position.symbol}[/]")
                    self._close_position(position, current_price, "stop_loss")
                elif position.should_take_profit(current_price):
                    console.print(f"[green]Take-profit: {position.symbol}[/]")
                    self._close_position(position, current_price, "take_profit")
                elif position.should_trailing_stop(current_price):
                    console.print(f"[yellow]Trailing stop: {position.symbol}[/]")
                    self._close_position(position, current_price, "trailing_stop")

            except Exception as e:
                console.print(f"[red]Error managing {position.symbol}: {e}[/]")

    def _close_position(self, position: Position, current_price: float, reason: str):
        result = self.exchange.close_position(position.symbol)
        if isinstance(result, dict) and result.get("error"):
            console.print(f"[red]   Close failed: {result['error']}[/]")
            return

        exit_price = float(result.get("exit_price") or current_price) if isinstance(result, dict) else current_price
        pnl = position.calculate_pnl(exit_price)
        notional_margin = position.size * position.entry_price / max(position.leverage, 1)
        pnl_pct = (pnl / notional_margin) * 100 if notional_margin else 0
        duration = int((datetime.utcnow() - position.opened_at).total_seconds())

        trade_result = TradeResult(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            leverage=position.leverage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_seconds=duration,
            exit_reason=reason,
            strategy=position.strategy,
        )
        self.risk.record_trade(trade_result)

        if pnl < 0:
            self.strategy.record_loss(datetime.utcnow())
        else:
            self.strategy.record_win()

        color = "green" if pnl >= 0 else "red"
        console.print(f"[{color}]   PnL: ${pnl:.2f} ({pnl_pct:.1f}%) — {reason}[/]")

    def _display_status(self):
        if self.cycle_count % 5 != 0:
            return

        account = self.exchange.get_account_state()
        stats = self.risk.get_stats()

        table = Table(title=f"Status (Cycle #{self.cycle_count})", show_header=True)
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

        pair_stats = stats.get("pair_stats", {})
        if pair_stats:
            table.add_row("", "")
            table.add_row("[bold]Pair Performance[/]", "")
            for symbol, ps in pair_stats.items():
                name = symbol.split("/")[0]
                table.add_row(
                    f"  {name}",
                    f"PnL=${ps['total_pnl']:.0f} | Sharpe={ps['sharpe']:.2f} | "
                    f"Weight={ps['weight']:.2f}x | Trades={ps['trades']}",
                )

        console.print(table)

    def _load_config(self, path: str) -> dict:
        config_path = Path(path)
        if not config_path.exists():
            return {}
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

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
            title="Session Ended",
        ))


def main():
    console.print("[bold green]WEEX AI Wars II — Trading Bot v8[/]")
    console.print("[dim]Press Ctrl+C to stop[/]\n")
    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()
