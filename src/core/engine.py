"""WEEX AI Wars II — Main Trading Engine v8.4

- HTF data, adaptive strategy scores
- State persistence across restarts
- Partial take-profit handling
- File logging
"""

import time
import yaml
import signal as sig
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
from ..utils.logger import setup_logger
from ..utils.state import save_state, load_state, DEFAULT_STATE_PATH
import numpy as np

console = Console()


class TradingEngine:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.exchange = ExchangeClient(self.config)
        self.strategy = CompositeStrategy(self.config)
        self.risk = RiskManager(self.config)
        self.logger = setup_logger(self.config)
        self.running = False
        self.cycle_count = 0
        self.state_path = Path(
            self.config.get("logging", {}).get("state_file", str(DEFAULT_STATE_PATH))
        )

        # Restore state
        state = load_state(self.state_path)
        if state:
            self.risk.load_state(state.get("risk") or {})
            lt = state.get("last_trade_time") or {}
            for k, v in lt.items():
                try:
                    self.strategy.last_trade_time[k] = datetime.fromisoformat(v.replace("Z", ""))
                except Exception:
                    pass
            self.logger.info("Restored bot state from %s", self.state_path)

        self.strategy.sync_scores_from_risk(self.risk)
        sig.signal(sig.SIGINT, self._shutdown)
        sig.signal(sig.SIGTERM, self._shutdown)

    def run(self):
        self.running = True
        symbols = self.config.get("trading", {}).get("symbols", ["BTC/USDT:USDT"])
        timeframe = self.config.get("trading", {}).get("timeframe", "1h")
        htf = self.config.get("trading", {}).get("higher_timeframe", "4h")
        lookback = self.config.get("trading", {}).get("lookback_periods", 100)
        htf_lookback = self.config.get("trading", {}).get("htf_lookback", 80)

        pure = self.config.get("competition", {}).get("pure_edge", False)
        console.print(Panel.fit(
            "[bold green]WEEX AI Wars II — Trading Bot v8.5[/]\n"
            f"Mode: [yellow]{self.config['trading']['mode']}[/] | "
            f"Profile: [cyan]{'pure_edge' if pure else 'competition'}[/]\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Timeframes: {timeframe} + {htf}\n"
            f"Max Drawdown: {self.risk.max_drawdown:.0%}\n"
            f"Risk/Trade: {self.risk.max_risk_per_trade:.1%}\n"
            f"Features: wick-MR, partial runners, adaptive weights, state save",
            title="Bot Started",
        ))
        self.logger.info(
            "Bot start mode=%s symbols=%s",
            self.config["trading"]["mode"],
            symbols,
        )

        leverage = self.config.get("trading", {}).get("default_leverage", 5)
        for symbol in symbols:
            self.exchange.set_leverage(symbol, leverage)

        while self.running:
            try:
                self.cycle_count += 1
                self._run_cycle(symbols, timeframe, htf, lookback, htf_lookback)
                self._display_status()
                if self.cycle_count % 3 == 0:
                    self._persist_state()

                sleep_time = 60 if timeframe == "1h" else 30
                for _ in range(sleep_time):
                    if not self.running:
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error in cycle: {e}[/]")
                self.logger.exception("Cycle error: %s", e)
                time.sleep(30)

        self._persist_state()
        self._cleanup()

    def _run_cycle(self, symbols, timeframe, htf, lookback, htf_lookback):
        account = self.exchange.get_account_state()
        self.strategy.sync_scores_from_risk(self.risk)

        can_trade, reason = self.risk.can_trade(account)
        if not can_trade:
            console.print(f"[yellow]Trading blocked: {reason}[/]")
            self.logger.warning("Trading blocked: %s", reason)
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

                self._execute_trade(signal, size, pair_weight)
                account = self.exchange.get_account_state()
                existing = [(p.symbol, p.side.value) for p in account.positions]

            except Exception as e:
                console.print(f"[red]Error analyzing {symbol}: {e}[/]")
                self.logger.exception("Analyze %s: %s", symbol, e)

    def _execute_trade(self, signal: Signal, size: float, pair_weight: float = 1.0):
        console.print(f"\n[bold cyan]Signal: {signal.side.value.upper()} {signal.symbol}[/]")
        console.print(f"   Strategy: {signal.strategy}")
        console.print(f"   Entry: ${signal.entry_price:.4f}")
        console.print(f"   Stop: ${signal.stop_loss:.4f}")
        console.print(f"   TP: ${signal.take_profit:.4f}")
        if signal.partial_take_profit:
            console.print(f"   Partial TP: ${signal.partial_take_profit:.4f} ({signal.partial_fraction:.0%})")
        console.print(f"   R:R = {signal.risk_reward_ratio:.2f}")
        console.print(f"   Strength: {signal.strength:.2f}")
        console.print(f"   Size: {size:.6f}")
        console.print(f"   Pair weight: {pair_weight:.2f}x | Strat weight: {self.risk.get_strategy_weight(signal.strategy):.2f}x")
        console.print(f"   Reason: {signal.reason}")

        self.logger.info(
            "SIGNAL %s %s %s str=%.2f size=%.6f SL=%.4f TP=%.4f | %s",
            signal.side.value, signal.symbol, signal.strategy,
            signal.strength, size, signal.stop_loss, signal.take_profit, signal.reason,
        )

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
            self.logger.error("Order failed: %s", result["error"])
            return

        # Attach partial TP metadata on paper positions
        if self.exchange.mode == "paper" and signal.symbol in self.exchange.paper_positions:
            pos = self.exchange.paper_positions[signal.symbol]
            pos.partial_take_profit = signal.partial_take_profit
            pos.partial_fraction = signal.partial_fraction
            pos.initial_size = size
            pos.strategy = signal.strategy

        # Live: stash partial levels in brackets
        if self.exchange.mode != "paper":
            br = self.exchange._local_brackets.get(signal.symbol, {})
            br["partial_take_profit"] = signal.partial_take_profit
            br["partial_fraction"] = signal.partial_fraction
            br["initial_size"] = size
            br["strategy"] = signal.strategy
            self.exchange._local_brackets[signal.symbol] = br

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

                # Restore partial metadata from brackets if needed
                if self.exchange.mode != "paper":
                    br = self.exchange._local_brackets.get(position.symbol, {})
                    if not position.partial_take_profit and br.get("partial_take_profit"):
                        position.partial_take_profit = br["partial_take_profit"]
                        position.partial_fraction = br.get("partial_fraction", 0.5)
                        position.initial_size = br.get("initial_size") or position.size
                        position.partial_taken = br.get("partial_taken", False)

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

                # Partial take-profit
                position, realized, closed = self.risk.apply_partial_tp(
                    position, current_price, atr
                )
                if realized is not None and closed > 0:
                    console.print(
                        f"[green]Partial TP: {position.symbol} closed {closed:.6f} "
                        f"PnL=${realized:.2f} — stop→BE[/]"
                    )
                    self.logger.info(
                        "PARTIAL_TP %s closed=%.6f pnl=%.2f remaining=%.6f",
                        position.symbol, closed, realized, position.size,
                    )
                    # Paper: reduce size in place; live would need reduce-only order
                    if self.exchange.mode == "paper":
                        if position.symbol in self.exchange.paper_positions:
                            if position.size <= 1e-12:
                                self.exchange.paper_positions.pop(position.symbol, None)
                            else:
                                self.exchange.paper_positions[position.symbol] = position
                                self.exchange.balance += realized
                    else:
                        try:
                            side = "sell" if position.side == Side.LONG else "buy"
                            self.exchange.exchange.create_order(
                                position.symbol, "market", side, closed,
                                params={"reduceOnly": True},
                            )
                            br = self.exchange._local_brackets.get(position.symbol, {})
                            br["partial_taken"] = True
                            br["stop_loss"] = position.stop_loss
                            self.exchange._local_brackets[position.symbol] = br
                        except Exception as e:
                            self.logger.error("Partial close failed: %s", e)
                    if position.size <= 1e-12:
                        continue

                position = self.risk.adjust_stops(position, current_price, atr)
                self.exchange.update_local_brackets(position)

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
                self.logger.exception("Manage %s: %s", position.symbol, e)

    def _close_position(self, position: Position, current_price: float, reason: str):
        result = self.exchange.close_position(position.symbol)
        if isinstance(result, dict) and result.get("error"):
            console.print(f"[red]   Close failed: {result['error']}[/]")
            return

        exit_price = (
            float(result.get("exit_price") or current_price)
            if isinstance(result, dict)
            else current_price
        )
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
        self.strategy.sync_scores_from_risk(self.risk)

        if pnl < 0:
            self.strategy.record_loss(datetime.utcnow())
        else:
            self.strategy.record_win()

        color = "green" if pnl >= 0 else "red"
        console.print(f"[{color}]   PnL: ${pnl:.2f} ({pnl_pct:.1f}%) — {reason}[/]")
        self.logger.info(
            "CLOSE %s %s pnl=%.2f reason=%s strategy=%s",
            position.symbol, position.side.value, pnl, reason, position.strategy,
        )
        self._persist_state()

    def _persist_state(self):
        try:
            lt = {
                k: (v.isoformat() if hasattr(v, "isoformat") else str(v))
                for k, v in self.strategy.last_trade_time.items()
            }
            save_state(
                self.state_path,
                {
                    "risk": self.risk.to_state(),
                    "last_trade_time": lt,
                    "cycle_count": self.cycle_count,
                },
            )
        except Exception as e:
            self.logger.warning("State save failed: %s", e)

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

        pair_stats = stats.get("pair_stats", {})
        if pair_stats:
            table.add_row("", "")
            table.add_row("[bold]Pairs[/]", "")
            for symbol, ps in pair_stats.items():
                name = symbol.split("/")[0]
                table.add_row(
                    f"  {name}",
                    f"PnL=${ps['total_pnl']:.0f} | W={ps['weight']:.2f}x | n={ps['trades']}",
                )

        strat = stats.get("strategy_stats", {})
        if strat:
            table.add_row("", "")
            table.add_row("[bold]Strategies[/]", "")
            for name, ss in strat.items():
                table.add_row(
                    f"  {name}",
                    f"PnL=${ss['total_pnl']:.0f} | W={ss['weight']:.2f}x | n={ss['trades']}",
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
        self.logger.info("Session ended stats=%s", stats)


def main():
    console.print("[bold green]WEEX AI Wars II — Trading Bot v8.5[/]")
    console.print("[dim]Press Ctrl+C to stop[/]\n")
    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()
