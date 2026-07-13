"""WEEX AI Wars II — Backtesting Engine v8

- Resamples 1h → 4h for real HTF confluence in strategy
- Strategy-level PnL breakdown
- Trailing / win-loss recorded from actual PnL
"""

import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.models import Candle, Side, Position, TradeResult
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
    strategy_stats: dict = field(default_factory=dict)
    final_capital: float = 0.0


def resample_to_htf(candles: list[Candle], hours: int = 4) -> list[Candle]:
    """Aggregate 1h candles into non-overlapping higher timeframe bars."""
    if not candles or hours < 1:
        return []
    out: list[Candle] = []
    for i in range(0, len(candles) - hours + 1, hours):
        chunk = candles[i : i + hours]
        out.append(
            Candle(
                timestamp=chunk[-1].timestamp,
                open=chunk[0].open,
                high=max(x.high for x in chunk),
                low=min(x.low for x in chunk),
                close=chunk[-1].close,
                volume=sum(x.volume for x in chunk),
            )
        )
    return out


def htf_window_for_index(all_htf: list[Candle], ts: datetime, max_bars: int = 80) -> list[Candle]:
    """HTF bars with timestamp <= current bar time."""
    if not all_htf:
        return []
    eligible = [c for c in all_htf if c.timestamp <= ts]
    return eligible[-max_bars:]


class Backtester:
    """Run strategy on historical data and measure performance."""

    def __init__(self, config: dict):
        self.config = config
        self.strategy = CompositeStrategy(config)
        self.initial_capital = config.get("backtest", {}).get("initial_capital", 10000)
        self.commission_rate = config.get("backtest", {}).get("commission_rate", 0.0006)
        self.slippage_pct = config.get("backtest", {}).get("slippage_pct", 0.001)

    def run(
        self,
        candles: list[Candle],
        symbol: str = "BTC/USDT:USDT",
        funding_rates: Optional[list[float]] = None,
        higher_tf_candles: Optional[list[Candle]] = None,
    ) -> BacktestResult:
        risk_config = dict(self.config.get("risk", {}))
        # Keep risk limits for realistic competition sim, but don't hard-kill entire run at 15%
        # unless config asks for it
        use_kill = self.config.get("backtest", {}).get("use_kill_switch", False)
        if not use_kill:
            risk_config["max_drawdown"] = 1.0
        risk_manager = RiskManager({**self.config, "risk": risk_config})
        risk_manager.peak_equity = self.initial_capital

        capital = self.initial_capital
        position: Optional[Position] = None
        trades: list[TradeResult] = []
        equity_curve = [capital]

        lookback = max(100, self.config.get("trading", {}).get("lookback_periods", 100))
        htf_all = higher_tf_candles or resample_to_htf(candles, 4)

        for i in range(lookback, len(candles)):
            current_candle = candles[i]
            current_price = current_candle.close

            if position:
                # Use high/low extremes within the bar
                if position.side == Side.LONG:
                    position.update_extremes(current_candle.high)
                else:
                    position.update_extremes(current_candle.low)
                position.update_extremes(current_price)

                atr = self._calculate_atr(candles, i, 14)
                # Mid-bar approx: adjust trail using close (conservative)
                risk_manager.adjust_stops(position, current_price, atr)

                hit_stop = False
                hit_tp = False
                hit_trail = False
                fill_price = current_price
                reason = ""

                if position.side == Side.LONG:
                    if position.stop_loss > 0 and current_candle.low <= position.stop_loss:
                        hit_stop = True
                        fill_price = position.stop_loss
                        reason = "stop_loss"
                    elif position.take_profit > 0 and current_candle.high >= position.take_profit:
                        hit_tp = True
                        fill_price = position.take_profit
                        reason = "take_profit"
                    elif position.trailing_stop and current_candle.low <= position.trailing_stop:
                        hit_trail = True
                        fill_price = position.trailing_stop
                        reason = "trailing_stop"
                else:
                    if position.stop_loss > 0 and current_candle.high >= position.stop_loss:
                        hit_stop = True
                        fill_price = position.stop_loss
                        reason = "stop_loss"
                    elif position.take_profit > 0 and current_candle.low <= position.take_profit:
                        hit_tp = True
                        fill_price = position.take_profit
                        reason = "take_profit"
                    elif position.trailing_stop and current_candle.high >= position.trailing_stop:
                        hit_trail = True
                        fill_price = position.trailing_stop
                        reason = "trailing_stop"

                if hit_stop or hit_tp or hit_trail:
                    trade = self._close_position(position, fill_price, reason, capital)
                    trades.append(trade)
                    capital += trade.pnl
                    risk_manager.record_trade(trade)
                    if trade.pnl < 0:
                        self.strategy.record_loss(current_candle.timestamp)
                    else:
                        self.strategy.record_win()
                    position = None

            if position is None and i < len(candles) - 1:
                # Optional competition risk gate
                class MockAccount:
                    equity = capital
                    positions = []
                    available_margin = capital
                    balance = capital

                can, _ = risk_manager.can_trade(MockAccount())
                if can:
                    window = candles[i - lookback + 1 : i + 1]
                    fr = funding_rates[i] if funding_rates else 0.0
                    htf_win = htf_window_for_index(htf_all, current_candle.timestamp, 80)
                    signal = self.strategy.analyze(
                        symbol, window, fr, higher_tf_candles=htf_win or None
                    )

                    if signal:
                        size = risk_manager.calculate_position_size(signal, MockAccount())
                        if size > 0:
                            slip = self.slippage_pct
                            if signal.side == Side.LONG:
                                entry_price = current_price * (1 + slip)
                            else:
                                entry_price = current_price * (1 - slip)

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
                            commission = size * entry_price * self.commission_rate
                            capital -= commission

            unrealized = position.calculate_pnl(current_price) if position else 0
            equity = capital + unrealized
            equity_curve.append(equity)
            if equity > risk_manager.peak_equity:
                risk_manager.peak_equity = equity

        if position:
            trade = self._close_position(position, candles[-1].close, "end_of_backtest", capital)
            trades.append(trade)
            capital += trade.pnl
            risk_manager.record_trade(trade)

        result = self._calculate_stats(trades, equity_curve)
        result.final_capital = capital
        return result

    def _close_position(
        self, position: Position, exit_price: float, reason: str, capital: float
    ) -> TradeResult:
        if position.side == Side.LONG:
            actual_exit = exit_price * (1 - self.slippage_pct)
        else:
            actual_exit = exit_price * (1 + self.slippage_pct)

        pnl = position.calculate_pnl(actual_exit)
        commission = position.size * actual_exit * self.commission_rate
        pnl -= commission

        margin = position.size * position.entry_price / max(position.leverage, 1)
        pnl_pct = (pnl / margin) * 100 if margin else 0

        return TradeResult(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=actual_exit,
            size=position.size,
            leverage=position.leverage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_seconds=3600,
            exit_reason=reason,
            strategy=position.strategy or "",
        )

    def _calculate_atr(self, candles: list[Candle], index: int, period: int) -> float:
        if index < period:
            return max(candles[index].high - candles[index].low, 1e-8)
        trs = []
        for i in range(index - period + 1, index + 1):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close if i > 0 else candles[i].open
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        return float(np.mean(trs))

    def _calculate_stats(self, trades: list[TradeResult], equity_curve: list) -> BacktestResult:
        strat = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
        for t in trades:
            key = t.strategy or "unknown"
            strat[key]["trades"] += 1
            strat[key]["pnl"] += t.pnl
            if t.pnl > 0:
                strat[key]["wins"] += 1

        if not trades:
            return BacktestResult(
                total_trades=0, wins=0, losses=0, win_rate=0,
                total_pnl=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, profit_factor=0, avg_trade_pnl=0,
                avg_win=0, avg_loss=0, max_consecutive_losses=0,
                best_trade=0, worst_trade=0, avg_trade_duration_hours=0,
                strategy_stats=dict(strat),
                final_capital=self.initial_capital,
            )

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / len(pnls)
        total_pnl = sum(pnls)

        eq = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        drawdown = np.where(peak > 0, (peak - eq) / peak, 0)
        max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0

        returns = np.diff(eq) / np.where(eq[:-1] == 0, 1, eq[:-1])
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(8760))
        else:
            sharpe = 0.0

        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            sortino = float(np.mean(returns) / np.std(negative_returns) * np.sqrt(8760))
        else:
            sortino = 0.0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / gross_loss

        max_consecutive = 0
        current_consecutive = 0
        for p in pnls:
            if p < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        avg_duration = float(np.mean([t.duration_seconds for t in trades]) / 3600)

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
            avg_trade_pnl=float(np.mean(pnls)),
            avg_win=float(np.mean(wins)) if wins else 0,
            avg_loss=float(np.mean(losses)) if losses else 0,
            max_consecutive_losses=max_consecutive,
            best_trade=max(pnls),
            worst_trade=min(pnls),
            avg_trade_duration_hours=avg_duration,
            trades=trades,
            equity_curve=list(eq),
            strategy_stats=dict(strat),
            final_capital=float(eq[-1]) if len(eq) else self.initial_capital,
        )


def print_backtest_results(result: BacktestResult):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    color = "green" if result.total_pnl > 0 else "red"
    console.print(Panel.fit(
        f"[bold {color}]Total PnL: ${result.total_pnl:.2f}[/]\n"
        f"Win Rate: {result.win_rate:.1%}\n"
        f"Total Trades: {result.total_trades}\n"
        f"Final Capital: ${result.final_capital:.2f}",
        title="Backtest Results",
    ))

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
    console.print(table)

    if result.strategy_stats:
        st = Table(title="By Strategy", show_header=True)
        st.add_column("Strategy", style="cyan")
        st.add_column("Trades")
        st.add_column("Wins")
        st.add_column("PnL")
        for name, s in sorted(result.strategy_stats.items(), key=lambda x: -x[1]["pnl"]):
            wr = s["wins"] / s["trades"] if s["trades"] else 0
            st.add_row(name, str(s["trades"]), f"{s['wins']} ({wr:.0%})", f"${s['pnl']:+.2f}")
        console.print(st)

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
