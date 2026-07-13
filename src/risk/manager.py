"""WEEX AI Wars II — Risk Management Engine v8

Improvements:
1. Signal.strength scales position size (keep-alive stays small)
2. Trailing stop only activates after configured profit threshold
3. Win-streak cooldown disabled by default (competition mode)
4. Safer zero-stop handling
5. Dynamic pair weights from recent Sharpe
"""

import numpy as np
from datetime import datetime
from typing import Optional

from ..core.models import Signal, Side, Position, AccountState, TradeResult


class RiskManager:
    """Multi-layered risk management for competition trading."""

    def __init__(self, config: dict):
        risk_config = config.get("risk", {})
        self.max_risk_per_trade = risk_config.get("max_risk_per_trade", 0.015)
        self.max_drawdown = risk_config.get("max_drawdown", 0.15)
        self.max_open_positions = risk_config.get("max_open_positions", 2)
        self.cooldown_after_losses = risk_config.get("cooldown_after_losses", 3)
        self.max_consecutive_losses = risk_config.get("max_consecutive_losses", 3)
        self.daily_loss_limit = risk_config.get("daily_loss_limit", 0.03)
        self.trailing_stop_activation = risk_config.get("trailing_stop_activation", 0.012)
        self.trailing_stop_distance = risk_config.get("trailing_stop_distance", 0.006)
        self.chandelier_atr_mult = risk_config.get("chandelier_atr_mult", 2.5)
        # Competition: do not pause winners unless explicitly enabled
        self.win_streak_cooldown = risk_config.get("win_streak_cooldown", False)
        self.max_consecutive_wins = risk_config.get("max_consecutive_wins", 8)
        self.cooldown_after_wins = risk_config.get("cooldown_after_wins", 1)

        sizing_config = config.get("sizing", {})
        self.sizing_method = sizing_config.get("method", "half_kelly")
        self.default_win_rate = sizing_config.get("default_win_rate", 0.50)
        self.min_position_usd = sizing_config.get("min_position_usd", 10)
        self.max_position_pct = sizing_config.get("max_position_pct", 0.20)

        self.pair_weights: dict[str, float] = {}
        self.pair_sharpes: dict[str, list[float]] = {}

        self.peak_equity = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trades_since_loss = 0
        self.trades_since_win = 0
        self.daily_pnl = 0.0
        self.daily_reset_date = datetime.utcnow().date()
        self.trade_history: list[TradeResult] = []
        self.is_killed = False

    def can_trade(self, account: AccountState) -> tuple[bool, str]:
        today = datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_pnl = 0.0
            self.daily_reset_date = today

        if account.equity > self.peak_equity:
            self.peak_equity = account.equity

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - account.equity) / self.peak_equity
            if drawdown >= self.max_drawdown:
                self.is_killed = True
                return False, f"KILL-SWITCH: Drawdown {drawdown:.1%} >= {self.max_drawdown:.1%}"

        if self.is_killed:
            return False, "Trading halted by kill-switch"

        if self.peak_equity > 0:
            daily_loss_pct = abs(self.daily_pnl) / self.peak_equity
            if self.daily_pnl < 0 and daily_loss_pct >= self.daily_loss_limit:
                return False, f"Daily loss limit reached: {daily_loss_pct:.1%}"

        if self.consecutive_losses >= self.max_consecutive_losses:
            if self.trades_since_loss < self.cooldown_after_losses:
                return (
                    False,
                    f"Cooldown: {self.trades_since_loss}/{self.cooldown_after_losses} "
                    f"after {self.consecutive_losses} losses",
                )
            self.consecutive_losses = 0

        if self.win_streak_cooldown and self.consecutive_wins >= self.max_consecutive_wins:
            if self.trades_since_win < self.cooldown_after_wins:
                return (
                    False,
                    f"Win streak cooldown: {self.trades_since_win}/{self.cooldown_after_wins}",
                )
            self.consecutive_wins = 0

        if len(account.positions) >= self.max_open_positions:
            return False, f"Max positions: {len(account.positions)}/{self.max_open_positions}"

        return True, "OK"

    def calculate_position_size(
        self,
        signal: Signal,
        account: AccountState,
        pair_weight: float = 1.0,
    ) -> float:
        if account.available_margin <= 0 or account.equity <= 0:
            return 0

        # Strength scales risk (keep-alive ~0.1–0.2 → small size)
        strength = max(0.05, min(1.0, signal.strength if signal.strength else 0.5))
        risk_amount = account.equity * self.max_risk_per_trade * strength * pair_weight

        stop_distance_usd = abs(signal.entry_price - signal.stop_loss)
        if stop_distance_usd <= 0:
            stop_distance_usd = signal.entry_price * 0.015

        amount = risk_amount / stop_distance_usd

        if self.sizing_method == "half_kelly":
            win_rate = self._estimate_win_rate()
            kelly = self._half_kelly(win_rate, signal.risk_reward_ratio)
            amount *= kelly

        max_amount = (account.equity * self.max_position_pct * strength) / signal.entry_price
        amount = min(amount, max_amount)

        max_margin_amount = (
            account.available_margin * 0.9 * max(signal.leverage, 1)
        ) / signal.entry_price
        amount = min(amount, max_margin_amount)

        if amount * signal.entry_price < self.min_position_usd:
            return 0

        return amount

    def adjust_stops(
        self,
        position: Position,
        current_price: float,
        atr: float,
    ) -> Position:
        """Breakeven + activation-gated chandelier trailing stop."""
        position.update_extremes(current_price)
        atr = atr if atr and atr > 0 else current_price * 0.01

        # Profit as fraction of entry (not leveraged)
        if position.side == Side.LONG:
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price

        risk = abs(position.entry_price - position.stop_loss) if position.stop_loss > 0 else atr * 1.5

        # Breakeven once 1R in profit
        if position.side == Side.LONG:
            if current_price >= position.entry_price + risk * 1.0:
                be = position.entry_price + atr * 0.15
                if position.stop_loss <= 0 or position.stop_loss < be:
                    position.stop_loss = be
        else:
            if current_price <= position.entry_price - risk * 1.0:
                be = position.entry_price - atr * 0.15
                if position.stop_loss <= 0 or position.stop_loss > be:
                    position.stop_loss = be

        # Trailing only after activation threshold
        if profit_pct >= self.trailing_stop_activation:
            if position.side == Side.LONG:
                # Prefer ATR chandelier; also respect fixed distance floor
                chandelier = position.highest_price - atr * self.chandelier_atr_mult
                pct_trail = current_price * (1 - self.trailing_stop_distance)
                trail = max(chandelier, pct_trail)
                if position.trailing_stop is None or trail > position.trailing_stop:
                    position.trailing_stop = trail
                if position.trailing_stop > position.stop_loss:
                    position.stop_loss = position.trailing_stop
            else:
                chandelier = position.lowest_price + atr * self.chandelier_atr_mult
                pct_trail = current_price * (1 + self.trailing_stop_distance)
                trail = min(chandelier, pct_trail)
                if position.trailing_stop is None or trail < position.trailing_stop:
                    position.trailing_stop = trail
                if position.stop_loss <= 0 or position.trailing_stop < position.stop_loss:
                    position.stop_loss = position.trailing_stop

        return position

    def update_pair_performance(self, symbol: str, pnl: float):
        if symbol not in self.pair_sharpes:
            self.pair_sharpes[symbol] = []
        self.pair_sharpes[symbol].append(pnl)
        if len(self.pair_sharpes[symbol]) > 20:
            self.pair_sharpes[symbol] = self.pair_sharpes[symbol][-20:]

    def get_pair_weight(self, symbol: str) -> float:
        if symbol not in self.pair_sharpes or len(self.pair_sharpes[symbol]) < 5:
            return 1.0
        pnls = np.array(self.pair_sharpes[symbol])
        if np.std(pnls) == 0:
            return 1.0
        sharpe = float(np.mean(pnls) / np.std(pnls))
        if sharpe > 1.0:
            return min(1.8, 1.0 + sharpe * 0.4)
        if sharpe < 0:
            return max(0.45, 1.0 + sharpe * 0.35)
        return 1.0

    def record_trade(self, result: TradeResult):
        self.trade_history.append(result)
        self.update_pair_performance(result.symbol, result.pnl)

        if result.pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.trades_since_loss = 0
            self.trades_since_win += 1
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.trades_since_win = 0
            self.trades_since_loss += 1

        self.daily_pnl += result.pnl

    def get_stats(self) -> dict:
        if not self.trade_history:
            return {
                "total_trades": 0, "win_rate": 0, "avg_pnl": 0,
                "max_drawdown": 0, "sharpe_ratio": 0,
                "consecutive_losses": self.consecutive_losses,
                "consecutive_wins": self.consecutive_wins,
                "total_pnl": 0, "wins": 0, "losses": 0,
                "daily_pnl": self.daily_pnl, "pair_stats": {},
            }

        pnls = [t.pnl for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_pnl = float(np.mean(pnls)) if pnls else 0

        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = float(np.mean(pnls) / np.std(pnls))
        else:
            sharpe = 0

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        base = peak + 10000
        drawdown = (peak - cumulative) / base
        max_dd = float(np.max(drawdown)) if len(drawdown) else 0

        pair_stats = {}
        for symbol, trade_pnls in self.pair_sharpes.items():
            if trade_pnls:
                arr = np.array(trade_pnls)
                pair_sharpe = float(np.mean(arr) / np.std(arr)) if np.std(arr) > 0 else 0
                pair_stats[symbol] = {
                    "trades": len(trade_pnls),
                    "total_pnl": sum(trade_pnls),
                    "sharpe": pair_sharpe,
                    "weight": self.get_pair_weight(symbol),
                }

        return {
            "total_trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_pnl": sum(pnls),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "daily_pnl": self.daily_pnl,
            "pair_stats": pair_stats,
        }

    def reset_kill_switch(self):
        self.is_killed = False

    def _estimate_win_rate(self) -> float:
        if len(self.trade_history) < 10:
            return self.default_win_rate
        recent = self.trade_history[-50:]
        wins = sum(1 for t in recent if t.pnl > 0)
        return wins / len(recent)

    def _half_kelly(self, win_rate: float, risk_reward: float) -> float:
        if risk_reward <= 0:
            return 0.5
        loss_rate = 1 - win_rate
        full_kelly = (win_rate * risk_reward - loss_rate) / risk_reward
        return max(0.15, min(1.0, full_kelly / 2))
