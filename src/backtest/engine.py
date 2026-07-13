"""WEEX AI Wars II — Backtesting Engine"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from ..core.models import Candle, Signal, Side, Position, TradeResult, MarketRegime
from ..strategies.composite import CompositeStrategy
from ..risk.manager import RiskManager


@dataclass
class BacktestResult:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    best_trade: float
    worst_trade: float
    avg_trade_duration_hours: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


class Backtester:
    """Run strategy on historical data and measure performance."""

    def __init__(self, config: dict):
        self.config = config
        self.strategy = CompositeStrategy(config)
        self.initial_capital = config.get("backtest", {}).get("initial_capital", 10000)
        self.commission_rate = config.get("backtest", {}).get("commission_rate", 0.0006)  # 0.06% taker
        self.slippage_pct = config.get("backtest", {}).get("slippage_pct", 0.001)  # 0.1%

    def run(
        self,
        candles: list[Candle],
        symbol: str = "BTC/USDT:USDT",
        funding_rates: Optional[list[float]] = None,
    ) -> BacktestResult:
        """Run backtest on candle data."""
        risk_config = self.config.get("risk", {})
        risk_config["max_drawdown"] = 1.0  # Disable kill-switch for backtest
        risk_manager = RiskManager({**self.config, "risk": risk_config})

        capital = self.initial_capital
        position: Optional[Position] = None
        trades: list[TradeResult] = []
        equity_curve = [capital]
        peak_equity = capital

        lookback = 100  # Need 100 candles for EMA100

        for i in range(lookback, len(candles)):
            current_candle = candles[i]
            current_price = current_candle.close

            # Update position PnL
            if position:
                position.update_extremes(current_price)

                # Check stop-loss (use high/low for realistic fill)
                if position.side == Side.LONG:
                    hit_stop = current_candle.low <= position.stop_loss
                else:
                    hit_stop = current_candle.high >= position.stop_loss

                if hit_stop:
                    fill_price = position.stop_loss
                    trade = self._close_position(position, fill_price, "stop_loss", capital)
                    trades.append(trade)
                    capital += trade.pnl
                    position = None
                    self.strategy.record_loss(current_candle.timestamp)

                # Check take-profit (use high/low)
                elif position.side == Side.LONG:
                    if current_candle.high >= position.take_profit:
                        trade = self._close_position(position, position.take_profit, "take_profit", capital)
                        trades.append(trade)
                        capital += trade.pnl
                        position = None
                        self.strategy.record_win()
                else:
                    if current_candle.low <= position.take_profit:
                        trade = self._close_position(position, position.take_profit, "take_profit", capital)
                        trades.append(trade)
                        capital += trade.pnl
                        position = None
                        self.strategy.record_win()

                # Check trailing stop
                if position and position.should_trailing_stop(current_price):
                    trade = self._close_position(position, current_price, "trailing_stop", capital)
                    trades.append(trade)
                    capital += trade.pnl
                    position = None
                    self.strategy.record_win()  # Trailing stop = in profit

                # Update trailing stop
                if position:
                    atr = self._calculate_atr(candles, i, 14)
                    risk_manager.adjust_stops(position, current_price, atr)

            # Generate signal if no position
            if position is None and i < len(candles) - 1:
                window = candles[i - lookback + 1 : i + 1]
                fr = funding_rates[i] if funding_rates else 0.0
                signal = self.strategy.analyze(symbol, window, fr)

                if signal:
                    # Calculate position size
                    class MockAccount:
                        equity = capital
                        positions = []
                        available_margin = capital
                        balance = capital

                    size = risk_manager.calculate_position_size(signal, MockAccount())

                    if size > 0:
                        # Apply slippage
                        entry_price = current_price * (1 + self.slippage_pct if signal.side == Side.LONG else 1 - self.slippage_pct)

                        # Adjust SL/TP relative to slipped entry
                        sl_delta = signal.stop_loss - signal.entry_price
                        tp_delta = signal.take_profit - signal.entry_price

                        position = Position(
                            symbol=symbol,
                            side=signal.side,
                            entry_price=entry_price,
                            size=size,
                            leverage=signal.leverage,
                            stop_loss=entry_price + sl_delta,
                            take_profit=entry_price + tp_delta,
                            highest_price=entry_price,
                            lowest_price=entry_price,
                            strategy=signal.strategy,
                        )

                        # Deduct commission
                        commission = size * entry_price * self.commission_rate
                        capital -= commission

            # Track equity
            unrealized = position.calculate_pnl(current_price) if position else 0
            equity = capital + unrealized
            equity_curve.append(equity)
            if equity > peak_equity:
                peak_equity = equity

        # Close any remaining position
        if position:
            trade = self._close_position(position, candles[-1].close, "end_of_backtest", capital)
            trades.append(trade)
            capital += trade.pnl

        return self._calculate_stats(trades, equity_curve)

    def _close_position(
        self, position: Position, exit_price: float, reason: str, capital: float
    ) -> TradeResult:
        """Close position and calculate final PnL."""
        # Apply slippage on exit
        if position.side == Side.LONG:
            actual_exit = exit_price * (1 - self.slippage_pct)
        else:
            actual_exit = exit_price * (1 + self.slippage_pct)

        pnl = position.calculate_pnl(actual_exit)

        # Deduct commission
        commission = position.size * actual_exit * self.commission_rate
        pnl -= commission

        pnl_pct = pnl / (position.size * position.entry_price / position.leverage) * 100
        duration = 3600  # Simplified (1 candle = 1 hour)

        return TradeResult(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=actual_exit,
            size=position.size,
            leverage=position.leverage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_seconds=duration,
            exit_reason=reason,
        )

    def _calculate_atr(self, candles: list[Candle], index: int, period: int) -> float:
        """Calculate ATR at a specific index."""
        if index < period:
            return candles[index].high - candles[index].low

        trs = []
        for i in range(index - period + 1, index + 1):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close if i > 0 else candles[i].open
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        return np.mean(trs)

    def _calculate_stats(self, trades: list[TradeResult], equity_curve: list) -> BacktestResult:
        """Calculate comprehensive backtest statistics."""
        if not trades:
            return BacktestResult(
                total_trades=0, wins=0, losses=0, win_rate=0,
                total_pnl=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, profit_factor=0, avg_trade_pnl=0,
                avg_win=0, avg_loss=0, max_consecutive_losses=0,
                best_trade=0, worst_trade=0, avg_trade_duration_hours=0,
            )

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        # Win rate
        win_rate = len(wins) / len(pnls) if pnls else 0

        # Total PnL
        total_pnl = sum(pnls)

        # Max drawdown from equity curve
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_drawdown = np.max(drawdown)

        # Sharpe ratio (annualized, assuming hourly candles)
        returns = np.diff(eq) / eq[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760)  # Annualized
        else:
            sharpe = 0

        # Sortino ratio (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            sortino = np.mean(returns) / np.std(negative_returns) * np.sqrt(8760)
        else:
            sortino = 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss

        # Consecutive losses
        max_consecutive = 0
        current_consecutive = 0
        for p in pnls:
            if p < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        # Average trade duration
        avg_duration = np.mean([t.duration_seconds for t in trades]) / 3600

        return BacktestResult(
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            avg_trade_pnl=np.mean(pnls),
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
            max_consecutive_losses=max_consecutive,
            best_trade=max(pnls),
            worst_trade=min(pnls),
            avg_trade_duration_hours=avg_duration,
            trades=trades,
            equity_curve=equity_curve,
        )


def print_backtest_results(result: BacktestResult):
    """Pretty print backtest results."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Summary panel
    color = "green" if result.total_pnl > 0 else "red"
    console.print(Panel.fit(
        f"[bold {color}]Total PnL: ${result.total_pnl:.2f}[/]\n"
        f"Win Rate: {result.win_rate:.1%}\n"
        f"Total Trades: {result.total_trades}",
        title="📊 Backtest Results",
    ))

    # Detailed table
    table = Table(title="Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Trades", str(result.total_trades))
    table.add_row("Wins / Losses", f"{result.wins} / {result.losses}")
    table.add_row("Win Rate", f"{result.win_rate:.1%}")
    table.add_row("Total PnL", f"${result.total_pnl:.2f}")
    table.add_row("Avg Trade PnL", f"${result.avg_trade_pnl:.2f}")
    table.add_row("Avg Win", f"${result.avg_win:.2f}")
    table.add_row("Avg Loss", f"${result.avg_loss:.2f}")
    table.add_row("Best Trade", f"${result.best_trade:.2f}")
    table.add_row("Worst Trade", f"${result.worst_trade:.2f}")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("Max Drawdown", f"{result.max_drawdown:.1%}")
    table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{result.sortino_ratio:.2f}")
    table.add_row("Max Consecutive Losses", str(result.max_consecutive_losses))
    table.add_row("Avg Trade Duration", f"{result.avg_trade_duration_hours:.1f}h")

    console.print(table)

    # Exit reason breakdown
    if result.trades:
        reasons = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        reason_table = Table(title="Exit Reasons", show_header=True)
        reason_table.add_column("Reason", style="cyan")
        reason_table.add_column("Count", style="white")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            reason_table.add_row(reason, str(count))
        console.print(reason_table)
