"""Multi-symbol portfolio backtester — shared capital & position limits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ..core.models import Candle, Side, Position, TradeResult, AccountState
from ..strategies.composite import CompositeStrategy
from ..risk.manager import RiskManager
from .engine import BacktestResult, resample_to_htf, htf_window_for_index


@dataclass
class SymbolSeries:
    symbol: str
    candles: list[Candle]
    funding: list[float]
    htf: list[Candle]
    # timestamp -> index
    index_by_ts: dict = field(default_factory=dict)


class PortfolioBacktester:
    """
    Align symbols on shared 1h timestamps.
    One capital pool, max_open_positions enforced.
    """

    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config.get("backtest", {}).get("initial_capital", 10000)
        self.commission_rate = config.get("backtest", {}).get("commission_rate", 0.0006)
        self.slippage_pct = config.get("backtest", {}).get("slippage_pct", 0.0005)

    def run(self, market: dict[str, dict]) -> BacktestResult:
        """
        market[symbol] = {candles, funding, htf?}
        """
        series: list[SymbolSeries] = []
        for symbol, data in market.items():
            candles = data["candles"]
            funding = data.get("funding") or [0.0] * len(candles)
            htf = data.get("htf") or resample_to_htf(candles, 4)
            idx = {c.timestamp: i for i, c in enumerate(candles)}
            series.append(SymbolSeries(symbol, candles, funding, htf, idx))

        if not series:
            return self._empty()

        # Shared timeline = intersection of hours present in all (or union of majority)
        ts_sets = [set(s.index_by_ts.keys()) for s in series]
        common = sorted(set.intersection(*ts_sets)) if len(ts_sets) > 1 else sorted(ts_sets[0])
        if len(common) < 150:
            # fall back to union sorted, skip missing
            common = sorted(set.union(*ts_sets))

        lookback = max(100, self.config.get("trading", {}).get("lookback_periods", 100))
        risk_cfg = dict(self.config.get("risk", {}))
        if not self.config.get("backtest", {}).get("use_kill_switch", False):
            risk_cfg["max_drawdown"] = 1.0
        risk = RiskManager({**self.config, "risk": risk_cfg})
        risk.peak_equity = self.initial_capital
        strategy = CompositeStrategy(self.config)

        capital = float(self.initial_capital)
        positions: dict[str, Position] = {}
        trades: list[TradeResult] = []
        equity_curve = [capital]

        # Need enough history before first trade
        start_i = 0
        for i, ts in enumerate(common):
            # check each series has lookback
            ok = True
            for s in series:
                if ts not in s.index_by_ts or s.index_by_ts[ts] < lookback:
                    ok = False
                    break
            if ok:
                start_i = i
                break

        for ti in range(start_i, len(common)):
            ts = common[ti]
            strategy.sync_scores_from_risk(risk)

            # --- manage positions ---
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                s = next(x for x in series if x.symbol == symbol)
                if ts not in s.index_by_ts:
                    continue
                ci = s.index_by_ts[ts]
                candle = s.candles[ci]
                price = candle.close
                atr = self._atr(s.candles, ci)

                # Partial TP (bar high/low)
                if risk.partial_tp_enabled and pos.partial_take_profit and not pos.partial_taken:
                    hit_partial = False
                    fill = pos.partial_take_profit
                    if pos.side == Side.LONG and candle.high >= pos.partial_take_profit:
                        hit_partial = True
                    elif pos.side == Side.SHORT and candle.low <= pos.partial_take_profit:
                        hit_partial = True
                    if hit_partial:
                        pos, realized, closed = risk.apply_partial_tp(pos, fill, atr)
                        if realized is not None:
                            # commission on partial
                            realized -= closed * fill * self.commission_rate
                            capital += realized
                            risk.daily_pnl += realized
                            risk.update_strategy_performance(pos.strategy, realized)
                            risk.update_pair_performance(symbol, realized)
                            positions[symbol] = pos
                            if pos.size <= 1e-12:
                                del positions[symbol]
                                continue

                pos = risk.adjust_stops(pos, price, atr)
                positions[symbol] = pos

                hit = False
                fill_price = price
                reason = ""
                if pos.side == Side.LONG:
                    if pos.stop_loss > 0 and candle.low <= pos.stop_loss:
                        hit, fill_price, reason = True, pos.stop_loss, "stop_loss"
                    elif pos.take_profit > 0 and candle.high >= pos.take_profit:
                        hit, fill_price, reason = True, pos.take_profit, "take_profit"
                    elif pos.trailing_stop and candle.low <= pos.trailing_stop:
                        hit, fill_price, reason = True, pos.trailing_stop, "trailing_stop"
                else:
                    if pos.stop_loss > 0 and candle.high >= pos.stop_loss:
                        hit, fill_price, reason = True, pos.stop_loss, "stop_loss"
                    elif pos.take_profit > 0 and candle.low <= pos.take_profit:
                        hit, fill_price, reason = True, pos.take_profit, "take_profit"
                    elif pos.trailing_stop and candle.high >= pos.trailing_stop:
                        hit, fill_price, reason = True, pos.trailing_stop, "trailing_stop"

                if hit:
                    trade = self._close(pos, fill_price, reason)
                    trades.append(trade)
                    capital += trade.pnl
                    bar_t = ts if isinstance(ts, datetime) else datetime.utcnow()
                    risk.record_trade(trade, now=bar_t)
                    if trade.pnl < 0:
                        strategy.record_loss(ts)
                    else:
                        strategy.record_win()
                    del positions[symbol]

            # --- equity ---
            unrealized = 0.0
            for symbol, pos in positions.items():
                s = next(x for x in series if x.symbol == symbol)
                if ts in s.index_by_ts:
                    unrealized += pos.calculate_pnl(s.candles[s.index_by_ts[ts]].close)
            equity = capital + unrealized
            equity_curve.append(equity)
            if equity > risk.peak_equity:
                risk.peak_equity = equity

            # --- entries ---
            account = AccountState(
                balance=capital,
                equity=equity,
                unrealized_pnl=unrealized,
                margin_used=sum(
                    p.size * p.entry_price / max(p.leverage, 1) for p in positions.values()
                ),
                available_margin=max(
                    0.0,
                    capital
                    - sum(p.size * p.entry_price / max(p.leverage, 1) for p in positions.values()),
                ),
                positions=list(positions.values()),
            )
            # Use candle time so cooldown works in backtest (not wall clock)
            bar_time = ts if isinstance(ts, datetime) else datetime.utcnow()
            can, _ = risk.can_trade(account, now=bar_time)
            if not can or ti >= len(common) - 2:
                continue

            existing = [(p.symbol, p.side.value) for p in positions.values()]
            # Prefer better pair weights first
            ordered = sorted(
                series,
                key=lambda s: risk.get_pair_weight(s.symbol),
                reverse=True,
            )
            for s in ordered:
                if s.symbol in positions:
                    continue
                if len(positions) >= risk.max_open_positions:
                    break
                if ts not in s.index_by_ts:
                    continue
                ci = s.index_by_ts[ts]
                if ci < lookback:
                    continue
                window = s.candles[ci - lookback + 1 : ci + 1]
                fr = s.funding[ci] if ci < len(s.funding) else 0.0
                htf_win = htf_window_for_index(s.htf, ts, 80)
                signal = strategy.analyze(
                    s.symbol, window, fr, existing, higher_tf_candles=htf_win or None
                )
                if not signal:
                    continue
                size = risk.calculate_position_size(signal, account, risk.get_pair_weight(s.symbol))
                if size <= 0:
                    continue

                price = window[-1].close
                slip = self.slippage_pct
                entry = price * (1 + slip if signal.side == Side.LONG else 1 - slip)
                sl_d = signal.stop_loss - signal.entry_price
                tp_d = signal.take_profit - signal.entry_price
                ptp = None
                if signal.partial_take_profit:
                    ptp_d = signal.partial_take_profit - signal.entry_price
                    ptp = entry + ptp_d

                pos = Position(
                    symbol=s.symbol,
                    side=signal.side,
                    entry_price=entry,
                    size=size,
                    leverage=signal.leverage,
                    stop_loss=entry + sl_d,
                    take_profit=entry + tp_d,
                    highest_price=entry,
                    lowest_price=entry,
                    strategy=signal.strategy,
                    partial_take_profit=ptp,
                    partial_fraction=signal.partial_fraction,
                    initial_size=size,
                )
                capital -= size * entry * self.commission_rate
                positions[s.symbol] = pos
                existing = [(p.symbol, p.side.value) for p in positions.values()]
                # refresh account available for next symbol same bar
                account = AccountState(
                    balance=capital,
                    equity=capital,
                    unrealized_pnl=0,
                    margin_used=sum(
                        p.size * p.entry_price / max(p.leverage, 1) for p in positions.values()
                    ),
                    available_margin=max(
                        0.0,
                        capital
                        - sum(
                            p.size * p.entry_price / max(p.leverage, 1)
                            for p in positions.values()
                        ),
                    ),
                    positions=list(positions.values()),
                )

        # close leftovers
        last_ts = common[-1]
        for symbol, pos in list(positions.items()):
            s = next(x for x in series if x.symbol == symbol)
            px = s.candles[-1].close
            trade = self._close(pos, px, "end_of_backtest")
            trades.append(trade)
            capital += trade.pnl
            risk.record_trade(trade)

        return self._stats(trades, equity_curve, capital)

    def _close(self, position: Position, exit_price: float, reason: str) -> TradeResult:
        if position.side == Side.LONG:
            actual = exit_price * (1 - self.slippage_pct)
        else:
            actual = exit_price * (1 + self.slippage_pct)
        pnl = position.calculate_pnl(actual)
        pnl -= position.size * actual * self.commission_rate
        margin = position.size * position.entry_price / max(position.leverage, 1)
        return TradeResult(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=actual,
            size=position.size,
            leverage=position.leverage,
            pnl=pnl,
            pnl_pct=(pnl / margin * 100) if margin else 0,
            duration_seconds=3600,
            exit_reason=reason,
            strategy=position.strategy or "",
        )

    def _atr(self, candles: list[Candle], index: int, period: int = 14) -> float:
        if index < period:
            return max(candles[index].high - candles[index].low, 1e-8)
        trs = []
        for i in range(index - period + 1, index + 1):
            h, l = candles[i].high, candles[i].low
            pc = candles[i - 1].close if i > 0 else candles[i].open
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return float(np.mean(trs))

    def _empty(self) -> BacktestResult:
        return BacktestResult(
            total_trades=0, wins=0, losses=0, win_rate=0, total_pnl=0,
            max_drawdown=0, sharpe_ratio=0, sortino_ratio=0, profit_factor=0,
            avg_trade_pnl=0, avg_win=0, avg_loss=0, max_consecutive_losses=0,
            best_trade=0, worst_trade=0, avg_trade_duration_hours=0,
            final_capital=self.initial_capital,
        )

    def _stats(self, trades, equity_curve, capital) -> BacktestResult:
        from .engine import Backtester
        bt = Backtester(self.config)
        result = bt._calculate_stats(trades, equity_curve)
        result.final_capital = capital
        return result
