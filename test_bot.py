"""Quick test to verify all modules work."""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.core.models import Candle, Signal, Side, Position, MarketRegime
from src.indicators.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, detect_regime, calculate_vwap,
    calculate_stochastic_rsi, calculate_ema, calculate_sma,
)
from src.risk.manager import RiskManager
from src.strategies.composite import CompositeStrategy

# Generate sample market data
np.random.seed(42)
closes = 100 + np.cumsum(np.random.randn(100) * 0.5)
highs = closes + np.random.rand(100) * 0.5
lows = closes - np.random.rand(100) * 0.5
volumes = np.random.rand(100) * 1000000

print("=== INDICATOR TESTS ===")

rsi = calculate_rsi(closes, 14)
print(f"RSI: last={rsi[-1]:.1f}, range=[{rsi.min():.1f}, {rsi.max():.1f}]")

macd, signal_line, hist = calculate_macd(closes)
print(f"MACD: last={macd[-1]:.3f}, signal={signal_line[-1]:.3f}")

bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(closes)
print(f"BB: upper={bb_upper[-1]:.2f}, mid={bb_mid[-1]:.2f}, lower={bb_lower[-1]:.2f}")

atr = calculate_atr(highs, lows, closes)
print(f"ATR: last={atr[-1]:.3f}")

adx = calculate_adx(highs, lows, closes)
print(f"ADX: last={adx[-1]:.1f}")

vwap = calculate_vwap(highs, lows, closes, volumes)
print(f"VWAP: last={vwap[-1]:.2f}")

stoch_k, stoch_d = calculate_stochastic_rsi(closes)
print(f"StochRSI: K={stoch_k[-1]:.1f}, D={stoch_d[-1]:.1f}")

regime = detect_regime(highs, lows, closes)
print(f"Regime: {regime}")

print("\n=== RISK MANAGER TEST ===")
config = {
    "risk": {"max_risk_per_trade": 0.02, "max_drawdown": 0.20, "max_open_positions": 3, "cooldown_after_losses": 5, "max_consecutive_losses": 3, "daily_loss_limit": 0.05, "trailing_stop_activation": 0.02, "trailing_stop_distance": 0.01},
    "sizing": {"method": "half_kelly", "default_win_rate": 0.55, "min_position_usd": 10, "max_position_pct": 0.25},
}
rm = RiskManager(config)

# Mock account
class MockAccount:
    equity = 10000
    positions = []
    available_margin = 10000
    balance = 10000

can, reason = rm.can_trade(MockAccount())
print(f"Can trade: {can}, reason: {reason}")

# Test position sizing
sig = Signal(
    symbol="BTC/USDT:USDT", side=Side.LONG, strength=0.7,
    strategy="trend_follow", entry_price=50000,
    stop_loss=49000, take_profit=52000, leverage=5,
    reason="test"
)
size = rm.calculate_position_size(sig, MockAccount())
print(f"Position size: {size:.6f} BTC (${size * 50000:.2f})")

print("\n=== STRATEGY TEST ===")
candles = [
    Candle(timestamp=None, open=float(closes[i]), high=float(highs[i]),
           low=float(lows[i]), close=float(closes[i]), volume=float(volumes[i]))
    for i in range(100)
]

strategy = CompositeStrategy({
    "strategy": {
        "trend_follow": {"enabled": True, "weight": 0.6, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bb_period": 20, "bb_std": 2.0, "adx_threshold": 25},
        "mean_reversion": {"enabled": True, "weight": 0.4, "rsi_period": 14, "rsi_overbought": 75, "rsi_oversold": 25, "bb_period": 20, "bb_std": 2.0},
    },
    "indicators": {"adx_period": 14},
    "trading": {"default_leverage": 5},
})

signal = strategy.analyze("BTC/USDT:USDT", candles, 0.0001)
if signal:
    print(f"Signal: {signal.side.value} {signal.symbol}")
    print(f"Strategy: {signal.strategy}")
    print(f"Strength: {signal.strength:.2f}")
    print(f"R:R: {signal.risk_reward_ratio:.1f}")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop: ${signal.stop_loss:.2f}")
    print(f"TP: ${signal.take_profit:.2f}")
    print(f"Reason: {signal.reason}")
else:
    print("No signal (normal - random data doesn't always trigger)")

print("\n=== POSITION TEST ===")
pos = Position(
    symbol="BTC/USDT:USDT", side=Side.LONG, entry_price=50000,
    size=0.1, leverage=5, stop_loss=49000, take_profit=52000,
    highest_price=50000, lowest_price=50000,
)
print(f"PnL at 51000: ${pos.calculate_pnl(51000):.2f}")
print(f"PnL at 49000: ${pos.calculate_pnl(49000):.2f}")
print(f"Stop loss at 48900: {pos.should_stop_loss(48900)}")
print(f"Take profit at 52100: {pos.should_take_profit(52100)}")

print("\n=== ALL TESTS PASSED ===")
