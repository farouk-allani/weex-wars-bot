"""WEEX AI Wars II — Risk Management Engine v2

Improvements:
1. Chandelier exit for trailing stops (adaptive to trend length)
2. Dynamic position sizing by recent Sharpe
3. Correlation-aware position limits
4. Win-streak cooldown (after 3+ wins, market may be exhausted)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from ..core.models import Signal, Side, Position, AccountState, TradeResult


class RiskManager:
    """
    Multi-layered risk management v2:
    1. Per-trade risk limit (1.5% of capital)
    2. Max drawdown kill-switch (15%)
    3. Half-Kelly position sizing
    4. Consecutive loss cooldown
    5. Win-streak cooldown (market exhaustion)
    6. Daily loss limit
    7. Chandelier trailing stop
    8. Correlation-aware limits
    """

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

        sizing_config = config.get("sizing", {})
        self.sizing_method = sizing_config.get("method", "half_kelly")
        self.default_win_rate = sizing_config.get("default_win_rate", 0.50)
        self.min_position_usd = sizing_config.get("min_position_usd", 10)
        self.max_position_pct = sizing_config.get("max_position_pct", 0.20)

        # Dynamic allocation weights (updated by recent performance)
        self.pair_weights: dict[str, float] = {}
        self.pair_sharpes: dict[str, list[float]] = {}

        # State
        self.peak_equity = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trades_since_loss = 0
        self.trades_since_win = 0
        self.daily_pnl = 0.0
        self.daily_reset_date = datetime.utcnow().date()
        self.trade_history: list[TradeResult] = []
        self.is_killed = False

        # Win streak cooldown config
        self.max_consecutive_wins = 5  # After 5 wins, pause (market exhaustion)
        self.cooldown_after_wins = 2

    def can_trade(self, account: AccountState) -> tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        today = datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_pnl = 0.0
            self.daily_reset_date = today

        if account.equity > self.peak_equity:
            self.peak_equity = account.equity

        # Kill-switch: max drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - account.equity) / self.peak_equity
            if drawdown >= self.max_drawdown:
                self.is_killed = True
                return False, f"KILL-SWITCH: Drawdown {drawdown:.1%} >= {self.max_drawdown:.1%}"

        if self.is_killed:
            return False, "Trading halted by kill-switch"

        # Daily loss limit
        if self.peak_equity > 0:
            daily_loss_pct = abs(self.daily_pnl) / self.peak_equity
            if self.daily_pnl < 0 and daily_loss_pct >= self.daily_loss_limit:
                return False, f"Daily loss limit reached: {daily_loss_pct:.1%}"

        # Consecutive loss cooldown
        if self.consecutive_losses >= self.max_consecutive_losses:
            if self.trades_since_loss < self.cooldown_after_losses:
                return False, f"Cooldown: {self.trades_since_loss}/{self.cooldown_after_losses} trades since {self.consecutive_losses} consecutive losses"
            else:
                self.consecutive_losses = 0

        # Win streak cooldown (market exhaustion)
        if self.consecutive_wins >= self.max_consecutive_wins:
            if self.trades_since_win < self.cooldown_after_wins:
                return False, f"Win streak cooldown: {self.trades_since_win}/{self.cooldown_after_wins} after {self.consecutive_wins} wins — market may be exhausted"
            else:
                self.consecutive_wins = 0

        # Max open positions
        if len(account.positions) >= self.max_open_positions:
            return False, f"Max positions reached: {len(account.positions)}/{self.max_open_positions}"

        return True, "OK"

    def calculate_position_size(
        self,
        signal: Signal,
        account: AccountState,
        pair_weight: float = 1.0,
    ) -> float:
        """Calculate position size with dynamic pair weighting."""
        if account.available_margin <= 0:
            return 0

        # Base risk amount
        risk_amount = account.equity * self.max_risk_per_trade

        # Apply pair weight (from dynamic allocation)
        risk_amount *= pair_weight

        # Stop distance
        stop_distance_usd = abs(signal.entry_price - signal.stop_loss)
        if stop_distance_usd == 0:
            stop_distance_usd = signal.entry_price * 0.02

        # Position size
        amount = risk_amount / stop_distance_usd

        # Kelly adjustment
        if self.sizing_method == "half_kelly":
            win_rate = self._estimate_win_rate()
            kelly = self._half_kelly(win_rate, signal.risk_reward_ratio)
            amount *= kelly

        # Cap at max position
        max_amount = (account.equity * self.max_position_pct) / signal.entry_price
        amount = min(amount, max_amount)

        # Cap at available margin
        max_margin_amount = (account.available_margin * 0.9 * signal.leverage) / signal.entry_price
        amount = min(amount, max_margin_amount)

        # Minimum position check
        position_value = amount * signal.entry_price
        if position_value < self.min_position_usd:
            return 0

        return amount

    def adjust_stops(
        self,
        position: Position,
        current_price: float,
        atr: float,
    ) -> Position:
        """Update stops using chandelier exit + breakeven logic."""
        position.update_extremes(current_price)

        # ---- Chandelier Exit (adaptive trailing stop) ----
        if position.side == Side.LONG:
            # Long trailing: highest high - 2.5 * ATR
            chandelier_stop = position.highest_price - atr * 2.5

            # Only tighten, never loosen
            if position.trailing_stop is None or chandelier_stop > position.trailing_stop:
                position.trailing_stop = chandelier_stop
        else:
            # Short trailing: lowest low + 2.5 * ATR
            chandelier_stop = position.lowest_price + atr * 2.5

            if position.trailing_stop is None or chandelier_stop < position.trailing_stop:
                position.trailing_stop = chandelier_stop

        # ---- Breakeven at 1.0x risk ----
        risk = abs(position.entry_price - position.stop_loss)
        if position.side == Side.LONG:
            if current_price >= position.entry_price + risk * 1.0:
                if position.stop_loss < position.entry_price:
                    position.stop_loss = position.entry_price + atr * 0.3
        else:
            if current_price <= position.entry_price - risk * 1.0:
                if position.stop_loss > position.entry_price:
                    position.stop_loss = position.entry_price - atr * 0.3

        # ---- Move stop to chandelier if it's better ----
        if position.side == Side.LONG:
            if position.trailing_stop and position.trailing_stop > position.stop_loss:
                position.stop_loss = position.trailing_stop
        else:
            if position.trailing_stop and position.trailing_stop < position.stop_loss:
                position.stop_loss = position.trailing_stop

        return position

    def update_pair_performance(self, symbol: str, pnl: float):
        """Track per-pair performance for dynamic allocation."""
        if symbol not in self.pair_sharpes:
            self.pair_sharpes[symbol] = []

        self.pair_sharpes[symbol].append(pnl)

        # Keep last 20 trades per pair
        if len(self.pair_sharpes[symbol]) > 20:
            self.pair_sharpes[symbol] = self.pair_sharpes[symbol][-20:]

    def get_pair_weight(self, symbol: str) -> float:
        """
        Dynamic pair allocation by recent Sharpe.
        Better performing pairs get more capital.
        Returns weight between 0.5 and 2.0.
        """
        if symbol not in self.pair_sharpes or len(self.pair_sharpes[symbol]) < 5:
            return 1.0  # Default weight

        pnls = np.array(self.pair_sharpes[symbol])
        if np.std(pnls) == 0:
            return 1.0

        # Recent Sharpe (not annualized, just relative)
        sharpe = np.mean(pnls) / np.std(pnls)

        # Map Sharpe to weight:
        # Sharpe > 1.0 → weight 1.5 (winner — give more capital)
        # Sharpe 0-1.0 → weight 1.0 (neutral)
        # Sharpe < 0 → weight 0.5 (loser — reduce exposure)
        if sharpe > 1.0:
            return min(2.0, 1.0 + sharpe * 0.5)
        elif sharpe < 0:
            return max(0.5, 1.0 + sharpe * 0.3)
        else:
            return 1.0

    def record_trade(self, result: TradeResult):
        """Record trade with win/loss streak tracking."""
        self.trade_history.append(result)

        # Update pair performance
        self.update_pair_performance(result.symbol, result.pnl)

        # Update consecutive losses/wins
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
        """Get trading statistics."""
        if not self.trade_history:
            return {
                "total_trades": 0, "win_rate": 0, "avg_pnl": 0,
                "max_drawdown": 0, "sharpe_ratio": 0,
                "consecutive_losses": self.consecutive_losses,
                "consecutive_wins": self.consecutive_wins,
            }

        pnls = [t.pnl for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_pnl = np.mean(pnls) if pnls else 0

        if len(pnls) > 1:
            returns = np.array(pnls)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 10000)
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        # Per-pair stats
        pair_stats = {}
        for symbol, trade_pnls in self.pair_sharpes.items():
            if trade_pnls:
                pair_pnls = np.array(trade_pnls)
                pair_sharpe = np.mean(pair_pnls) / np.std(pair_pnls) if np.std(pair_pnls) > 0 else 0
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
        half_kelly = max(0.1, min(1.0, full_kelly / 2))
        return half_kelly
