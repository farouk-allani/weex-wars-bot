"""WEEX AI Wars II — Risk Management Engine"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from ..core.models import Signal, Side, Position, AccountState, TradeResult


class RiskManager:
    """
    Multi-layered risk management:
    1. Per-trade risk limit (2% of capital)
    2. Max drawdown kill-switch (20%)
    3. Half-Kelly position sizing
    4. Consecutive loss cooldown
    5. Daily loss limit
    6. Trailing stop management
    """

    def __init__(self, config: dict):
        risk_config = config.get("risk", {})
        self.max_risk_per_trade = risk_config.get("max_risk_per_trade", 0.02)
        self.max_drawdown = risk_config.get("max_drawdown", 0.20)
        self.max_open_positions = risk_config.get("max_open_positions", 3)
        self.cooldown_after_losses = risk_config.get("cooldown_after_losses", 5)
        self.max_consecutive_losses = risk_config.get("max_consecutive_losses", 3)
        self.daily_loss_limit = risk_config.get("daily_loss_limit", 0.05)
        self.trailing_stop_activation = risk_config.get("trailing_stop_activation", 0.02)
        self.trailing_stop_distance = risk_config.get("trailing_stop_distance", 0.01)

        sizing_config = config.get("sizing", {})
        self.sizing_method = sizing_config.get("method", "half_kelly")
        self.default_win_rate = sizing_config.get("default_win_rate", 0.55)
        self.min_position_usd = sizing_config.get("min_position_usd", 10)
        self.max_position_pct = sizing_config.get("max_position_pct", 0.25)

        # State
        self.peak_equity = 0.0
        self.consecutive_losses = 0
        self.trades_since_loss = 0
        self.daily_pnl = 0.0
        self.daily_reset_date = datetime.utcnow().date()
        self.trade_history: list[TradeResult] = []
        self.is_killed = False

    def can_trade(self, account: AccountState) -> tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        # Reset daily PnL if new day
        today = datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_pnl = 0.0
            self.daily_reset_date = today

        # Update peak equity
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

        # Max open positions
        if len(account.positions) >= self.max_open_positions:
            return False, f"Max positions reached: {len(account.positions)}/{self.max_open_positions}"

        return True, "OK"

    def calculate_position_size(
        self,
        signal: Signal,
        account: AccountState,
    ) -> float:
        """Calculate position size using Half-Kelly or fixed risk."""
        if account.available_margin <= 0:
            return 0

        # Calculate risk in USD
        risk_amount = account.equity * self.max_risk_per_trade

        # Distance from entry to stop-loss as percentage
        stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        if stop_distance == 0:
            stop_distance = 0.02  # Default 2% stop

        # Position size = risk / (stop_distance * leverage)
        position_value = risk_amount / (stop_distance * signal.leverage)

        # Apply Kelly criterion adjustment
        if self.sizing_method == "half_kelly":
            win_rate = self._estimate_win_rate()
            kelly = self._half_kelly(win_rate, signal.risk_reward_ratio)
            position_value *= kelly

        # Cap at max position percentage
        max_value = account.equity * self.max_position_pct
        position_value = min(position_value, max_value)

        # Cap at available margin
        position_value = min(position_value, account.available_margin * 0.9)

        # Minimum position check
        if position_value < self.min_position_usd:
            return 0

        # Convert to base currency amount
        if signal.entry_price > 0:
            amount = position_value / signal.entry_price
        else:
            amount = 0

        return amount

    def adjust_stops(
        self,
        position: Position,
        current_price: float,
        atr: float,
    ) -> Position:
        """Update stop-loss and take-profit based on price action."""
        position.update_extremes(current_price)

        # Activate trailing stop
        if position.side == Side.LONG:
            profit_pct = (current_price - position.entry_price) / position.entry_price
            if profit_pct >= self.trailing_stop_activation:
                trailing = current_price - atr * 2
                if position.trailing_stop is None or trailing > position.trailing_stop:
                    position.trailing_stop = trailing
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price
            if profit_pct >= self.trailing_stop_activation:
                trailing = current_price + atr * 2
                if position.trailing_stop is None or trailing < position.trailing_stop:
                    position.trailing_stop = trailing

        # Move stop to breakeven at 1.5x risk
        risk = abs(position.entry_price - position.stop_loss)
        if position.side == Side.LONG:
            if current_price >= position.entry_price + risk * 1.5:
                if position.stop_loss < position.entry_price:
                    position.stop_loss = position.entry_price + atr * 0.5
        else:
            if current_price <= position.entry_price - risk * 1.5:
                if position.stop_loss > position.entry_price:
                    position.stop_loss = position.entry_price - atr * 0.5

        return position

    def record_trade(self, result: TradeResult):
        """Record a completed trade for statistics."""
        self.trade_history.append(result)

        # Update consecutive losses
        if result.pnl < 0:
            self.consecutive_losses += 1
            self.trades_since_loss = 0
        else:
            self.consecutive_losses = 0
            self.trades_since_loss += 1

        # Update daily PnL
        self.daily_pnl += result.pnl

    def get_stats(self) -> dict:
        """Get trading statistics."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "consecutive_losses": self.consecutive_losses,
            }

        pnls = [t.pnl for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_pnl = np.mean(pnls) if pnls else 0

        # Calculate Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns = np.array(pnls)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Max drawdown from trade history
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 10000)  # + starting capital
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

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
            "daily_pnl": self.daily_pnl,
        }

    def reset_kill_switch(self):
        """Manually reset the kill-switch (use with caution)."""
        self.is_killed = False

    def _estimate_win_rate(self) -> float:
        """Estimate win rate from recent trades."""
        if len(self.trade_history) < 10:
            return self.default_win_rate

        recent = self.trade_history[-50:]
        wins = sum(1 for t in recent if t.pnl > 0)
        return wins / len(recent)

    def _half_kelly(self, win_rate: float, risk_reward: float) -> float:
        """Half-Kelly criterion for position sizing."""
        if risk_reward <= 0:
            return 0.5

        # Full Kelly: f* = (p * b - q) / b
        # where p = win rate, q = loss rate, b = risk/reward ratio
        loss_rate = 1 - win_rate
        full_kelly = (win_rate * risk_reward - loss_rate) / risk_reward

        # Half Kelly for safety
        half_kelly = max(0.1, min(1.0, full_kelly / 2))

        return half_kelly
