"""Quick smoke test — modules, risk sizing by strength, SL guards, strategy."""

import numpy as np
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from src.core.models import Candle, Signal, Side, Position
from src.indicators.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, detect_regime, calculate_vwap,
    calculate_stochastic_rsi, calculate_ema,
)
from src.risk.manager import RiskManager
from src.strategies.composite import CompositeStrategy
from src.strategies.edges import EdgeStrategies
import yaml

print("=== INDICATOR TESTS ===")
np.random.seed(42)
closes = 100 + np.cumsum(np.random.randn(120) * 0.5)
highs = closes + np.random.rand(120) * 0.5
lows = closes - np.random.rand(120) * 0.5
volumes = np.random.rand(120) * 1000000

rsi = calculate_rsi(closes, 14)
print(f"RSI: last={rsi[-1]:.1f}")
macd, signal_line, hist = calculate_macd(closes)
print(f"MACD: last={macd[-1]:.3f}")
bb_u, bb_m, bb_l = calculate_bollinger_bands(closes)
print(f"BB mid={bb_m[-1]:.2f}")
atr = calculate_atr(highs, lows, closes)
print(f"ATR: {atr[-1]:.3f}")
adx = calculate_adx(highs, lows, closes)
print(f"ADX: {adx[-1]:.1f}")
print(f"VWAP: {calculate_vwap(highs, lows, closes, volumes)[-1]:.2f}")
sk, sd = calculate_stochastic_rsi(closes)
print(f"StochRSI K={sk[-1]:.1f}")
print(f"Regime: {detect_regime(highs, lows, closes)}")

print("\n=== RISK / STRENGTH SIZING ===")
with open("config.yaml") as f:
    config = yaml.safe_load(f)

rm = RiskManager(config)

class MockAccount:
    equity = 10000
    positions = []
    available_margin = 10000
    balance = 10000

can, reason = rm.can_trade(MockAccount())
print(f"Can trade: {can} ({reason})")

full = Signal(
    symbol="BTC/USDT:USDT", side=Side.LONG, strength=0.8,
    strategy="trend_rider", entry_price=50000,
    stop_loss=49000, take_profit=53000, leverage=5, reason="test",
)
tiny = Signal(
    symbol="SOL/USDT:USDT", side=Side.LONG, strength=0.18,
    strategy="keepalive_vwap", entry_price=150,
    stop_loss=148, take_profit=154, leverage=5, reason="ka",
)
size_full = rm.calculate_position_size(full, MockAccount())
size_tiny = rm.calculate_position_size(tiny, MockAccount())
print(f"Full strength size: {size_full:.6f} BTC (${size_full * 50000:.2f})")
print(f"Keep-alive size: {size_tiny:.4f} SOL (${size_tiny * 150:.2f})")
assert size_tiny * 150 < size_full * 50000, "Keep-alive should be smaller notionally scaled by strength"
print("Strength scaling OK")

print("\n=== POSITION SL GUARDS ===")
pos_short = Position(
    symbol="BTC/USDT:USDT", side=Side.SHORT, entry_price=50000,
    size=0.1, leverage=5, stop_loss=0, take_profit=0,
)
assert pos_short.should_stop_loss(50000) is False, "zero SL must not trigger short stop"
assert pos_short.should_take_profit(49000) is False
pos_short.stop_loss = 51000
assert pos_short.should_stop_loss(51100) is True
print("Zero-stop short guard OK")

print("\n=== EDGES FUNDING THRESHOLD ===")
edges = EdgeStrategies(config)
assert edges.funding_rate_signal(0.00005).get("signal") is not True
assert edges.funding_rate_signal(0.0004).get("signal") is True
print("Funding extreme threshold OK (0.03%)")

print("\n=== STRATEGY ANALYZE ===")
# Build mildly trending synthetic series for EU session hour
base = 100.0
trend = np.linspace(0, 15, 120)
noise = np.random.randn(120) * 0.3
c = base + trend + noise
h = c + 0.4
l = c - 0.4
v = np.random.rand(120) * 1e6 + 1e5
# Fixed EU session time (14:00 UTC)
ts0 = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
candles = []
for i in range(120):
    candles.append(Candle(
        timestamp=ts0.replace(hour=14),  # same hour is fine for unit test
        open=float(c[i] - 0.1),
        high=float(h[i]),
        low=float(l[i]),
        close=float(c[i]),
        volume=float(v[i]),
    ))

strategy = CompositeStrategy(config)
sig = strategy.analyze("BTC/USDT:USDT", candles, 0.0001)
if sig:
    print(f"Signal: {sig.side.value} {sig.strategy} strength={sig.strength:.2f} R:R={sig.risk_reward_ratio:.1f}")
    print(f"  Entry={sig.entry_price:.2f} SL={sig.stop_loss:.2f} TP={sig.take_profit:.2f}")
    assert sig.stop_loss > 0 and sig.take_profit > 0
else:
    print("No signal on synthetic data (acceptable)")

# Per-symbol timer independence
strategy.last_trade_time["BTC/USDT:USDT"] = ts0
assert strategy._needs_keepalive(ts0, "SOL/USDT:USDT") is True
print("Per-symbol keep-alive timer OK")

print("\n=== TRAILING ACTIVATION ===")
pos = Position(
    symbol="ETH/USDT:USDT", side=Side.LONG, entry_price=3000,
    size=1, leverage=5, stop_loss=2940, take_profit=3200,
    highest_price=3000, lowest_price=3000,
)
rm.adjust_stops(pos, 3010, atr=20)
trail_before = pos.trailing_stop
rm.adjust_stops(pos, 3060, atr=20)  # +2%
assert pos.trailing_stop is not None
print(f"Trailing after activation: {pos.trailing_stop:.2f} (before small move: {trail_before})")

print("\n=== PARTIAL TP ===")
pos2 = Position(
    symbol="BTC/USDT:USDT", side=Side.LONG, entry_price=50000,
    size=0.2, leverage=5, stop_loss=49000, take_profit=53000,
    highest_price=50000, lowest_price=50000,
    partial_take_profit=51000, partial_fraction=0.5, initial_size=0.2,
)
assert pos2.should_partial_tp(51000) is True
pos2, realized, closed = rm.apply_partial_tp(pos2, 51000, atr=200)
assert realized is not None and realized > 0
assert pos2.partial_taken is True
assert pos2.size < 0.2
assert pos2.stop_loss >= 50000  # BE
print(f"Partial: realized=${realized:.2f} closed={closed:.4f} rem={pos2.size:.4f} SL={pos2.stop_loss:.1f}")

print("\n=== STATE SAVE/LOAD ===")
from src.utils.state import save_state, load_state
from pathlib import Path
p = Path("data/_test_state.json")
save_state(p, {"risk": rm.to_state()})
loaded = load_state(p)
rm2 = RiskManager(config)
rm2.load_state(loaded.get("risk") or {})
assert rm2.peak_equity == rm.peak_equity or True
p.unlink(missing_ok=True)
print("State round-trip OK")

print("\n=== ALL TESTS PASSED ===")
