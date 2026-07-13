"""WEEX AI Wars II Trading Bot — Data Models"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class Signal:
    symbol: str
    side: Side
    strength: float  # 0.0 to 1.0 — scales position size
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # Optional scale-out level (e.g. 1R) — bank partial, trail rest
    partial_take_profit: Optional[float] = None
    partial_fraction: float = 0.5

    @property
    def risk_reward_ratio(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    size: float  # in contracts or base currency
    leverage: int
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    strategy: str = ""
    exchange_sl_set: bool = False
    exchange_tp_set: bool = False
    partial_take_profit: Optional[float] = None
    partial_fraction: float = 0.5
    partial_taken: bool = False
    initial_size: float = 0.0

    def update_extremes(self, price: float):
        if price > self.highest_price:
            self.highest_price = price
        if price < self.lowest_price:
            self.lowest_price = price

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate PnL in USD. Futures PnL = price_diff * size."""
        if self.side == Side.LONG:
            return (current_price - self.entry_price) * self.size
        return (self.entry_price - current_price) * self.size

    def should_stop_loss(self, current_price: float) -> bool:
        # stop_loss <= 0 means not set — never force-close shorts at 0
        if self.stop_loss is None or self.stop_loss <= 0:
            return False
        if self.side == Side.LONG:
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        if self.take_profit is None or self.take_profit <= 0:
            return False
        if self.side == Side.LONG:
            return current_price >= self.take_profit
        return current_price <= self.take_profit

    def should_trailing_stop(self, current_price: float) -> bool:
        if self.trailing_stop is None or self.trailing_stop <= 0:
            return False
        if self.side == Side.LONG:
            return current_price <= self.trailing_stop
        return current_price >= self.trailing_stop

    def should_partial_tp(self, current_price: float) -> bool:
        if self.partial_taken or not self.partial_take_profit or self.partial_take_profit <= 0:
            return False
        if self.side == Side.LONG:
            return current_price >= self.partial_take_profit
        return current_price <= self.partial_take_profit


@dataclass
class TradeResult:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    pnl: float
    pnl_pct: float
    duration_seconds: int
    exit_reason: str
    strategy: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccountState:
    balance: float
    equity: float
    unrealized_pnl: float
    margin_used: float
    available_margin: float
    positions: list = field(default_factory=list)

    @property
    def margin_ratio(self) -> float:
        return self.margin_used / self.equity if self.equity > 0 else 0
