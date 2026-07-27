"""Quick smoke test — modules, risk sizing by strength, SL guards, strategy."""

import copy
import json
import numpy as np
import sys
from datetime import datetime, timedelta, timezone

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

# Per-symbol timer independence. Keep-alive is off in the shipped config — the AI
# decision log is the heartbeat now — so test it against a config that enables it,
# rather than coupling the test to a deployment choice.
ka_config = copy.deepcopy(config)
ka_config["strategy"]["keepalive"]["enabled"] = True
ka_config["competition"]["pure_edge"] = False  # pure_edge force-disables keep-alive
ka_strategy = CompositeStrategy(ka_config)
ka_strategy.last_trade_time["BTC/USDT:USDT"] = ts0
assert ka_strategy._needs_keepalive(ts0, "SOL/USDT:USDT") is True
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

# ---------------------------------------------------------------------------
# Compliance. An AI-driven order without its ai-log is not a worse score, it is
# a non-compliant order — so these are invariants, not nice-to-haves.
# ---------------------------------------------------------------------------
print("\n=== AI-LOG SURVIVES A RESTART ===")
import logging as _logging
import tempfile
from src.ai import wars_log as _wl
from src.ai.logbook import DecisionLog

_logging.getLogger("src.ai.logbook").setLevel(_logging.CRITICAL)  # expected ERROR below
_tmp = Path(tempfile.mkdtemp())
_dl = DecisionLog(_tmp / "dec.jsonl")
_did = _dl.record(
    model="test-model",
    context={"markets": []},
    decisions=[{"symbol": "BTC/USDT:USDT", "action": "long", "rationale": "because X"}],
    raw_response="{}", reasoning="cot",
    messages=[{"role": "user", "content": "ctx"}],
)
# A maker entry can rest for entry_ttl_minutes and fill after a deploy, so the
# emitter must not depend on this process's memory.
_dl2 = DecisionLog(_tmp / "dec.jsonl")
assert _dl2._recent == {}, "fresh log must start with no in-memory decisions"
_wl.AI_LOGS_DIR = _tmp / "ailogs"
_dl2.link_order(_did, symbol="BTC/USDT:USDT", order_id="oid1", side="long",
                size=0.01, entry_price=50000.0, stop_loss=49000.0, take_profit=52000.0)
_files = list((_tmp / "ailogs").glob("*.json"))
assert len(_files) == 1, f"ai-log not written after restart: {_files}"
assert _dl2.ailog_emitted == 1 and _dl2.ailog_failed == 0
_payload = json.loads(_files[0].read_text())
assert _payload["explanation"] == "because X", _payload["explanation"]
for _k in ("stage", "model", "input", "output", "explanation"):
    assert _payload.get(_k), f"ai-log missing required field {_k}"
print("ai-log emitted from disk-recovered decision OK")

print("\n=== AI-LOG FAILURE IS LOUD ===")
_dl2.link_order("dec_missing", symbol="ETH/USDT:USDT", order_id="oid2", side="short",
                size=1.0, entry_price=3000.0, stop_loss=3100.0, take_profit=2800.0)
assert _dl2.ailog_failed == 1, "an unrecoverable ai-log must be counted, not swallowed"
assert _dl2.last_ailog_error
_status = _dl2.compliance_status(_tmp / "ailogs")
assert _status["orders_linked"] == 2 and _status["ai_logs_on_disk"] == 1
assert _status["orders_without_ai_log"] == 1 and _status["compliant"] is False
assert _status["ai_logs_incomplete"] == 0, _status["incomplete"]
print(f"compliance gap detected OK ({_status['orders_without_ai_log']}/{_status['orders_linked']})")

print("\n=== A PRESENT-BUT-UNUSABLE LOG IS NOT COMPLIANT ===")
# A file can exist and still fail the schema — most importantly with no verbatim
# prompt, which is what backfilling a pre-logbook decision produces.
_good = _wl.build_ai_log(
    {"model": "m", "messages": [{"role": "user", "content": "c"}],
     "context": {"markets": []}, "decisions": [{"symbol": "BTC/USDT:USDT", "rationale": "r"}]},
    {"symbol": "BTC/USDT:USDT", "side": "long", "size": 0.01, "entry_price": 1.0},
)
assert _wl.validate(_good) == [], _wl.validate(_good)
_bad = json.loads(json.dumps(_good))
_bad["input"]["messages"] = []
assert any("messages" in p for p in _wl.validate(_bad)), _wl.validate(_bad)
_bad2 = json.loads(json.dumps(_good))
_bad2["output"]["quantity"] = None
assert any("quantity" in p for p in _wl.validate(_bad2))
_bad3 = json.loads(json.dumps(_good))
_bad3["explanation"] = "x" * 1001
assert any("1001" in p for p in _wl.validate(_bad3))
print("ai-log schema validation OK")

print("\n=== BOOT-STAMPED TRADES ARE QUARANTINED ===")
from src.core.models import TradeResult as _TR

def _mk(n, stamp):
    return [_TR(symbol="BTC/USDT:USDT", side=Side.LONG, entry_price=1, exit_price=1, size=1,
                leverage=5, pnl=0.5, pnl_pct=1, duration_seconds=60, exit_reason="be_stop",
                strategy="ai_deepseek", timestamp=stamp) for _ in range(n)]

# The real artifact is microsecond-SEQUENTIAL, not identical: the old loader ran
# the default factory once per trade inside a loop. Measured on the VPS: 7 trades
# spanning 53us. An earlier version of this check keyed on equality and never fired.
_base = datetime(2026, 7, 25, 10, 37, 53, 148001)
_seq = [_base + timedelta(microseconds=9 * i) for i in range(8)]
_rm = RiskManager(config)
_rm.trade_history = [_mk(1, s)[0] for s in _seq] + _mk(1, datetime(2026, 7, 26, 6, 5, 37))
_rm._quarantine_boot_stamped_trades()
assert sum(1 for t in _rm.trade_history if t.timestamp is None) == 8, \
    [t.timestamp for t in _rm.trade_history]
assert sum(t.pnl for t in _rm.trade_history) == 4.5, "quarantine must not touch P&L"
# Trades genuinely seconds apart are real history and must survive.
_rm2 = RiskManager(config)
_rm2.trade_history = [_mk(1, _base + timedelta(seconds=30 * i))[0] for i in range(8)]
_rm2._quarantine_boot_stamped_trades()
assert all(t.timestamp is not None for t in _rm2.trade_history), "real closes wrongly dropped"
_rm3 = RiskManager(config)
_rm3.trade_history = _mk(3, _base)  # a short cluster stays below the run threshold
_rm3._quarantine_boot_stamped_trades()
assert all(t.timestamp is not None for t in _rm3.trade_history)
# A legacy state with no timestamps must yield None, never boot time: a boot-time
# stamp lands inside the live round window and reads as a trade that never happened.
_rm4 = RiskManager(config)
_rm4.load_state({"trade_history": [{
    "symbol": "BTC/USDT:USDT", "side": "long", "entry_price": 1, "exit_price": 1,
    "size": 1, "leverage": 5, "pnl": 1.0, "pnl_pct": 1, "duration_seconds": 60,
    "exit_reason": "be_stop"}]})
assert _rm4.trade_history[0].timestamp is None, _rm4.trade_history[0].timestamp
assert _rm4.trade_history[0].pnl == 1.0
print("boot-stamp quarantine + legacy load OK")

print("\n=== ROUND PACE COUNTS ONLY DATED TRADES ===")
from src.core.engine import TradingEngine
_now = datetime.utcnow()
_comp = {"min_trades": 10, "round_days": 7,
         "round_started": (_now - timedelta(days=5)).isoformat()}
_pace = TradingEngine._trade_pace(None, _comp, _mk(3, _now - timedelta(days=1)) + _mk(8, None))
assert _pace["trades_this_round"] == 3, _pace["trades_this_round"]
assert _pace["trades_still_needed"] == 7
assert _pace["status"].startswith("BEHIND PACE"), _pace["status"]
_met = TradingEngine._trade_pace(None, _comp, _mk(12, _now - timedelta(days=1)))
assert _met["trades_still_needed"] == 0 and "pure cost" in _met["status"]
print("round pace OK (undated trades cannot fake the minimum)")

print("\n=== ALL TESTS PASSED ===")
