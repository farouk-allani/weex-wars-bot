"""WEEX AI Wars II — Risk Management Engine v8.4

- Strength + pair + strategy adaptive sizing
- Partial take-profit support
- State serialize/restore for restarts
- Strategy-level performance weights
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Any

from ..core.models import Signal, Side, Position, AccountState, TradeResult


class RiskManager:
    def __init__(self, config: dict):
        risk_config = config.get("risk", {})
        self.max_risk_per_trade = risk_config.get("max_risk_per_trade", 0.015)
        self.max_drawdown = risk_config.get("max_drawdown", 0.15)
        self.max_open_positions = risk_config.get("max_open_positions", 2)
        self.cooldown_after_losses = risk_config.get("cooldown_after_losses", 3)
        self.max_consecutive_losses = risk_config.get("max_consecutive_losses", 3)
        # Time-based cooldown (hours) — avoids deadlock of "need trades to resume"
        self.cooldown_hours = float(risk_config.get("cooldown_hours", 6))
        self.daily_loss_limit = risk_config.get("daily_loss_limit", 0.03)
        self.trailing_stop_activation = risk_config.get("trailing_stop_activation", 0.012)
        self.trailing_stop_distance = risk_config.get("trailing_stop_distance", 0.006)
        self.chandelier_atr_mult = risk_config.get("chandelier_atr_mult", 2.5)
        self.win_streak_cooldown = risk_config.get("win_streak_cooldown", False)
        self.max_consecutive_wins = risk_config.get("max_consecutive_wins", 8)
        self.cooldown_after_wins = risk_config.get("cooldown_after_wins", 1)
        self.partial_tp_enabled = risk_config.get("partial_tp_enabled", True)
        self.partial_be_buffer_atr = risk_config.get("partial_be_buffer_atr", 0.1)
        # Majors move together. Two shorts across BTC/SOL is one directional bet at
        # double size, not two independent positions, so cap same-side exposure.
        self.max_same_side_positions = int(
            risk_config.get("max_same_side_positions", 1)
        )

        sizing_config = config.get("sizing", {})
        self.sizing_method = sizing_config.get("method", "half_kelly")
        self.default_win_rate = sizing_config.get("default_win_rate", 0.50)
        self.min_position_usd = sizing_config.get("min_position_usd", 10)
        self.max_position_pct = sizing_config.get("max_position_pct", 0.20)

        self.pair_weights: dict[str, float] = {}
        self.pair_sharpes: dict[str, list[float]] = {}
        self.strategy_pnls: dict[str, list[float]] = {}

        self.peak_equity = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trades_since_loss = 0
        self.trades_since_win = 0
        self.daily_pnl = 0.0
        self.daily_reset_date = datetime.utcnow().date()
        self.trade_history: list[TradeResult] = []
        self.is_killed = False
        self.cooldown_until: Optional[datetime] = None

    def can_trade(
        self,
        account: AccountState,
        now: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        now = now or datetime.utcnow()
        today = now.date() if hasattr(now, "date") else datetime.utcnow().date()
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

        # Time-based loss cooldown (fixed deadlock: old logic needed wins to resume)
        if self.cooldown_until is not None:
            if now < self.cooldown_until:
                return False, f"Loss cooldown until {self.cooldown_until.isoformat()}"
            self.cooldown_until = None
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

    def can_open(self, signal: Signal, account: AccountState) -> tuple[bool, str]:
        """Per-signal gate, on top of the account-level can_trade() checks."""
        if self.max_same_side_positions <= 0:
            return True, "OK"
        same_side = sum(1 for p in account.positions if p.side == signal.side)
        if same_side >= self.max_same_side_positions:
            return (
                False,
                f"Correlated exposure: already {same_side} "
                f"{signal.side.value} position(s), max {self.max_same_side_positions}",
            )
        return True, "OK"

    def get_strategy_weight(self, strategy: str) -> float:
        """Adaptive weight 0.5–1.4 from recent strategy PnL."""
        if not strategy or strategy not in self.strategy_pnls:
            return 1.0
        pnls = self.strategy_pnls[strategy]
        if len(pnls) < 3:
            return 1.0
        recent = pnls[-12:]
        total = sum(recent)
        wins = sum(1 for p in recent if p > 0)
        wr = wins / len(recent)
        if total > 0 and wr >= 0.5:
            return min(1.4, 1.0 + total / (abs(total) + 50) * 0.5)
        if total < 0 and wr < 0.4:
            return max(0.5, 0.75 + wr * 0.3)
        return 1.0

    def calculate_position_size(
        self,
        signal: Signal,
        account: AccountState,
        pair_weight: float = 1.0,
    ) -> float:
        if account.available_margin <= 0 or account.equity <= 0:
            return 0

        strength = max(0.05, min(1.0, signal.strength if signal.strength else 0.5))
        strat_w = self.get_strategy_weight(signal.strategy)
        risk_amount = (
            account.equity
            * self.max_risk_per_trade
            * strength
            * pair_weight
            * strat_w
        )

        stop_distance_usd = abs(signal.entry_price - signal.stop_loss)
        if stop_distance_usd <= 0:
            stop_distance_usd = signal.entry_price * 0.015

        amount = risk_amount / stop_distance_usd

        if self.sizing_method == "half_kelly":
            win_rate = self._estimate_win_rate(signal.strategy)
            kelly = self._half_kelly(win_rate, signal.risk_reward_ratio)
            amount *= kelly

        max_amount = (
            account.equity * self.max_position_pct * strength * strat_w
        ) / signal.entry_price
        amount = min(amount, max_amount)

        max_margin_amount = (
            account.available_margin * 0.9 * max(signal.leverage, 1)
        ) / signal.entry_price
        amount = min(amount, max_margin_amount)

        if amount * signal.entry_price < self.min_position_usd:
            return 0

        return amount

    def apply_partial_tp(
        self,
        position: Position,
        current_price: float,
        atr: float = 0.0,
    ) -> tuple[Position, Optional[float], float]:
        """
        If partial TP hit, reduce size and bank PnL on closed fraction.
        Returns (position, realized_pnl_or_None, closed_size).
        """
        if not self.partial_tp_enabled or not position.should_partial_tp(current_price):
            return position, None, 0.0

        frac = max(0.1, min(0.8, position.partial_fraction or 0.5))
        base_size = position.initial_size or position.size
        close_size = base_size * frac
        if close_size <= 0 or close_size >= position.size:
            close_size = position.size * frac

        # Realized on closed slice
        if position.side == Side.LONG:
            realized = (current_price - position.entry_price) * close_size
        else:
            realized = (position.entry_price - current_price) * close_size

        position.size = max(0.0, position.size - close_size)
        position.partial_taken = True

        # Move stop to breakeven (+ tiny buffer)
        buf = (atr or position.entry_price * 0.001) * self.partial_be_buffer_atr
        if position.side == Side.LONG:
            be = position.entry_price + buf
            if position.stop_loss < be:
                position.stop_loss = be
        else:
            be = position.entry_price - buf
            if position.stop_loss <= 0 or position.stop_loss > be:
                position.stop_loss = be

        return position, realized, close_size

    def adjust_stops(
        self,
        position: Position,
        current_price: float,
        atr: float,
    ) -> Position:
        position.update_extremes(current_price)
        atr = atr if atr and atr > 0 else current_price * 0.01

        if position.side == Side.LONG:
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price

        risk = (
            abs(position.entry_price - position.stop_loss)
            if position.stop_loss > 0
            else atr * 1.5
        )

        # Faster BE after partial taken
        be_trigger = 0.7 if position.partial_taken else 1.0
        if position.side == Side.LONG:
            if current_price >= position.entry_price + risk * be_trigger:
                be = position.entry_price + atr * 0.12
                if position.stop_loss <= 0 or position.stop_loss < be:
                    position.stop_loss = be
        else:
            if current_price <= position.entry_price - risk * be_trigger:
                be = position.entry_price - atr * 0.12
                if position.stop_loss <= 0 or position.stop_loss > be:
                    position.stop_loss = be

        # Earlier + tighter trail after partial (protect runner)
        act = self.trailing_stop_activation * (0.55 if position.partial_taken else 1.0)
        chand_mult = self.chandelier_atr_mult * (0.7 if position.partial_taken else 1.0)
        trail_dist = self.trailing_stop_distance * (0.75 if position.partial_taken else 1.0)
        if profit_pct >= act:
            if position.side == Side.LONG:
                chandelier = position.highest_price - atr * chand_mult
                pct_trail = current_price * (1 - trail_dist)
                trail = max(chandelier, pct_trail)
                if position.trailing_stop is None or trail > position.trailing_stop:
                    position.trailing_stop = trail
                if position.trailing_stop > position.stop_loss:
                    position.stop_loss = position.trailing_stop
            else:
                chandelier = position.lowest_price + atr * chand_mult
                pct_trail = current_price * (1 + trail_dist)
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

    def update_strategy_performance(self, strategy: str, pnl: float):
        if not strategy:
            return
        if strategy not in self.strategy_pnls:
            self.strategy_pnls[strategy] = []
        self.strategy_pnls[strategy].append(pnl)
        if len(self.strategy_pnls[strategy]) > 30:
            self.strategy_pnls[strategy] = self.strategy_pnls[strategy][-30:]

    def get_pair_weight(self, symbol: str) -> float:
        base = symbol.split("/")[0] if "/" in symbol else symbol
        # Prior from backtests until enough live samples
        prior = {"BTC": 1.15, "ETH": 0.6, "SOL": 0.9}.get(base, 1.0)

        if symbol not in self.pair_sharpes or len(self.pair_sharpes[symbol]) < 4:
            return prior
        pnls = np.array(self.pair_sharpes[symbol])
        if np.std(pnls) == 0:
            return prior
        sharpe = float(np.mean(pnls) / np.std(pnls))
        if sharpe > 1.0:
            return min(1.9, prior * (1.0 + sharpe * 0.25))
        if sharpe < 0:
            return max(0.35, prior * (0.7 + sharpe * 0.15))
        return prior

    @staticmethod
    def is_keepalive(strategy: str) -> bool:
        return (strategy or "").startswith("keepalive")

    def record_partial(self, pnl: float):
        """Bank a partial scale-out immediately.

        Only daily_pnl moves here: it gates the daily loss limit and must reflect
        cash that is already realized. Pair/strategy performance is credited once,
        at close, from the full round-trip PnL.
        """
        self.daily_pnl += pnl

    def record_trade(self, result: TradeResult, now: Optional[datetime] = None):
        self.trade_history.append(result)
        now = now or datetime.utcnow()

        is_keepalive = self.is_keepalive(result.strategy)

        # Keep-alive trades are token-sized heartbeats, not signals. Letting them
        # steer pair weights or Kelly win-rate would size real trades off noise.
        if not is_keepalive:
            self.update_pair_performance(result.symbol, result.pnl)
        self.update_strategy_performance(result.strategy, result.pnl)

        if result.pnl < 0:
            # Keep-alive losses should not trigger full portfolio cooldown
            if not is_keepalive:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.trades_since_loss = 0
                self.trades_since_win += 1
                if self.consecutive_losses >= self.max_consecutive_losses:
                    self.cooldown_until = now + timedelta(hours=self.cooldown_hours)
                    self.consecutive_losses = 0
            else:
                self.trades_since_win += 1
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.trades_since_win = 0
            self.trades_since_loss += 1
            self.cooldown_until = None

        # banked_pnl already hit daily_pnl via record_partial()
        self.daily_pnl += result.pnl - result.banked_pnl

    def get_stats(self) -> dict:
        if not self.trade_history:
            return {
                "total_trades": 0, "win_rate": 0, "avg_pnl": 0,
                "max_drawdown": 0, "sharpe_ratio": 0,
                "consecutive_losses": self.consecutive_losses,
                "consecutive_wins": self.consecutive_wins,
                "total_pnl": 0, "wins": 0, "losses": 0,
                "daily_pnl": self.daily_pnl, "pair_stats": {},
                "strategy_stats": {},
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

        strategy_stats = {}
        for name, sp in self.strategy_pnls.items():
            strategy_stats[name] = {
                "trades": len(sp),
                "total_pnl": sum(sp),
                "weight": self.get_strategy_weight(name),
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
            "strategy_stats": strategy_stats,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "peak_equity": self.peak_equity,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "trades_since_loss": self.trades_since_loss,
            "trades_since_win": self.trades_since_win,
            "daily_pnl": self.daily_pnl,
            "daily_reset_date": str(self.daily_reset_date),
            "is_killed": self.is_killed,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "pair_sharpes": self.pair_sharpes,
            "strategy_pnls": self.strategy_pnls,
            "trade_history": [
                {
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "leverage": t.leverage,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "duration_seconds": t.duration_seconds,
                    "exit_reason": t.exit_reason,
                    "strategy": t.strategy,
                    "banked_pnl": t.banked_pnl,
                    "fees": t.fees,
                }
                for t in self.trade_history[-200:]
            ],
        }

    def load_state(self, state: dict[str, Any]):
        if not state:
            return
        self.peak_equity = float(state.get("peak_equity") or 0)
        self.consecutive_losses = int(state.get("consecutive_losses") or 0)
        self.consecutive_wins = int(state.get("consecutive_wins") or 0)
        self.trades_since_loss = int(state.get("trades_since_loss") or 0)
        self.trades_since_win = int(state.get("trades_since_win") or 0)
        self.daily_pnl = float(state.get("daily_pnl") or 0)
        self.is_killed = bool(state.get("is_killed") or False)

        # daily_pnl is only meaningful together with the day it belongs to. Without
        # this, a restart adopts today's date while keeping yesterday's PnL, and a
        # losing yesterday keeps eating into today's loss limit.
        saved_day = state.get("daily_reset_date")
        today = datetime.utcnow().date()
        try:
            self.daily_reset_date = (
                datetime.fromisoformat(str(saved_day)).date() if saved_day else today
            )
        except Exception:
            self.daily_reset_date = today
        if self.daily_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_reset_date = today
        self.pair_sharpes = state.get("pair_sharpes") or {}
        self.strategy_pnls = state.get("strategy_pnls") or {}
        cu = state.get("cooldown_until")
        if cu:
            try:
                self.cooldown_until = datetime.fromisoformat(str(cu).replace("Z", ""))
            except Exception:
                self.cooldown_until = None
        else:
            self.cooldown_until = None
        # Rebuild minimal trade history for win-rate estimates
        self.trade_history = []
        for raw in state.get("trade_history") or []:
            try:
                self.trade_history.append(
                    TradeResult(
                        symbol=raw["symbol"],
                        side=Side(raw["side"]),
                        entry_price=float(raw["entry_price"]),
                        exit_price=float(raw["exit_price"]),
                        size=float(raw["size"]),
                        leverage=int(raw["leverage"]),
                        pnl=float(raw["pnl"]),
                        pnl_pct=float(raw.get("pnl_pct") or 0),
                        duration_seconds=int(raw.get("duration_seconds") or 0),
                        exit_reason=raw.get("exit_reason") or "",
                        strategy=raw.get("strategy") or "",
                        banked_pnl=float(raw.get("banked_pnl") or 0),
                        fees=float(raw.get("fees") or 0),
                    )
                )
            except Exception:
                continue

    def reset_kill_switch(self):
        self.is_killed = False

    def _estimate_win_rate(self, strategy: str = "") -> float:
        if strategy and strategy in self.strategy_pnls and len(self.strategy_pnls[strategy]) >= 5:
            recent = self.strategy_pnls[strategy][-30:]
            return sum(1 for p in recent if p > 0) / len(recent)
        real = [t for t in self.trade_history if not self.is_keepalive(t.strategy)]
        if len(real) < 10:
            return self.default_win_rate
        recent = real[-50:]
        wins = sum(1 for t in recent if t.pnl > 0)
        return wins / len(recent)

    def _half_kelly(self, win_rate: float, risk_reward: float) -> float:
        if risk_reward <= 0:
            return 0.5
        loss_rate = 1 - win_rate
        full_kelly = (win_rate * risk_reward - loss_rate) / risk_reward
        return max(0.15, min(1.0, full_kelly / 2))
