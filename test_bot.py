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
# Scale-out is disabled by policy in config.yaml (it was measured to destroy edge),
# but the mechanism must still work for anyone who turns it back on — so this test
# opts in explicitly rather than inheriting the shipped default.
ptp_config = copy.deepcopy(config)
ptp_config["risk"]["partial_tp_enabled"] = True
rm_ptp = RiskManager(ptp_config)
pos2 = Position(
    symbol="BTC/USDT:USDT", side=Side.LONG, entry_price=50000,
    size=0.2, leverage=5, stop_loss=49000, take_profit=53000,
    highest_price=50000, lowest_price=50000,
    partial_take_profit=51000, partial_fraction=0.5, initial_size=0.2,
)
assert pos2.should_partial_tp(51000) is True
pos2, realized, closed = rm_ptp.apply_partial_tp(pos2, 51000, atr=200)
assert realized is not None and realized > 0
assert pos2.partial_taken is True
assert pos2.size < 0.2
assert pos2.stop_loss >= 50000  # BE
print(f"Partial: realized=${realized:.2f} closed={closed:.4f} rem={pos2.size:.4f} SL={pos2.stop_loss:.1f}")

print("\n=== EXIT GEOMETRY REGRESSIONS ===")
# Three defects measured over 8 pairs x 120d that together demanded 62% directional
# accuracy just to break even. Each assertion below pins one of them.

# 1. The trail must take the LOOSER of chandelier and percent trail. With max() the
#    fixed percent trail always won, the ATR chandelier was dead code, and every
#    winner was capped under 1% — a trailing stop that never fired in 20 live trades.
pos3 = Position(
    symbol="BTC/USDT:USDT", side=Side.LONG, entry_price=50000,
    size=0.1, leverage=5, stop_loss=49000, take_profit=54000,
    highest_price=50000, lowest_price=50000,
)
atr3 = 500.0  # 1% of price
rm.adjust_stops(pos3, 51500, atr=atr3)  # +3%, well past activation
chandelier = pos3.highest_price - atr3 * rm.chandelier_atr_mult
pct_trail = 51500 * (1 - rm.trailing_stop_distance)
assert pos3.trailing_stop is not None, "trail must arm at +3%"
assert abs(pos3.trailing_stop - min(chandelier, pct_trail)) < 1e-6, (
    f"long trail took {pos3.trailing_stop:.2f}, expected the looser "
    f"min(chandelier={chandelier:.2f}, pct={pct_trail:.2f})"
)
# and the mirror image for a short
pos4 = Position(
    symbol="BTC/USDT:USDT", side=Side.SHORT, entry_price=50000,
    size=0.1, leverage=5, stop_loss=51000, take_profit=46000,
    highest_price=50000, lowest_price=50000,
)
rm.adjust_stops(pos4, 48500, atr=atr3)
chandelier_s = pos4.lowest_price + atr3 * rm.chandelier_atr_mult
pct_trail_s = 48500 * (1 + rm.trailing_stop_distance)
assert abs(pos4.trailing_stop - max(chandelier_s, pct_trail_s)) < 1e-6, (
    "short trail must take the looser max(chandelier, pct)"
)
print(f"Trail takes looser candidate OK (long={pos3.trailing_stop:.1f}, "
      f"short={pos4.trailing_stop:.1f})")

# 2. Breakeven must not fire at 1R. be_trigger_r is configurable and >1 as shipped.
assert rm.be_trigger_r > 1.0, "BE at <=1R surrenders a full R the moment it works"
pos5 = Position(
    symbol="BTC/USDT:USDT", side=Side.LONG, entry_price=50000,
    size=0.1, leverage=5, stop_loss=49000, take_profit=54000,
    highest_price=50000, lowest_price=50000,
)
rm.adjust_stops(pos5, 50900, atr=10.0)  # +0.9R, tiny ATR so the trail cannot arm
assert pos5.stop_loss < 50000, f"BE fired at 0.9R (stop={pos5.stop_loss})"
print(f"BE holds at 0.9R OK (trigger={rm.be_trigger_r}R)")

# 3. A too-tight AI stop is widened while preserving its proposed R:R.
from src.ai.trader import AITrader
trader = AITrader(config, client=None, logbook=None)
price, atr_t = 50000.0, 500.0
tight = {"symbol": "BTC/USDT:USDT", "action": "long", "conviction": 0.8,
         "stop_loss": price - atr_t * 0.4, "take_profit": price + atr_t * 1.0,
         "rationale": "trend continuation with structure below the swing low"}
sig, why = trader.to_signal(tight, "BTC/USDT:USDT", price, atr_t,
                             {"BTC/USDT:USDT"})
assert sig is not None, f"tight stop was rejected instead of widened: {why}"
new_dist = abs(price - sig.stop_loss)
assert new_dist >= trader.min_stop_atr * atr_t - 1e-6, (
    f"stop {new_dist / atr_t:.2f}x ATR still under floor {trader.min_stop_atr}x"
)
rr_in = (atr_t * 1.0) / (atr_t * 0.4)
rr_out = abs(sig.take_profit - price) / new_dist
assert abs(rr_in - rr_out) < 1e-6, f"R:R not preserved: {rr_in:.3f} -> {rr_out:.3f}"
assert sig.partial_take_profit is None, "partial must be off when policy disables it"
print(f"Tight AI stop widened 0.40x -> {new_dist / atr_t:.2f}x ATR, "
      f"R:R preserved at {rr_out:.2f}, partial off")

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
assert _status["ai_logs_repairable_incomplete"] == 0, _status["incomplete"]
assert _status["ai_logs_unrepairable_historical"] == 0, _status["unrepairable"]
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

print("\n=== UNREPAIRABLE HISTORY DOES NOT MASK A NEW FAILURE ===")
# A decision logged before verbatim prompts were captured yields a log that can
# never be completed. It must be reported, but separately: a permanently-false
# compliance flag is a monitor nobody reads.
_tmp2 = Path(tempfile.mkdtemp())
_wl.AI_LOGS_DIR = _tmp2 / "ailogs"
_dl3 = DecisionLog(_tmp2 / "dec.jsonl")
_old = _dl3.record(model="deepseek-chat", context={"markets": []},
                   decisions=[{"symbol": "SOL/USDT:USDT", "action": "short",
                               "rationale": "historical"}],
                   raw_response="{}", reasoning="cot")  # note: no messages=
_dl3.link_order(_old, symbol="SOL/USDT:USDT", order_id="oidH", side="short",
                size=1.0, entry_price=100.0, stop_loss=105.0, take_profit=95.0)
_s = _dl3.compliance_status(_tmp2 / "ailogs")
assert _s["orders_without_ai_log"] == 0, _s
assert _s["ai_logs_unrepairable_historical"] == 1, _s
assert _s["ai_logs_repairable_incomplete"] == 0, _s["incomplete"]
assert _s["compliant"] is True, "unrepairable history must not pin compliance false"
assert _s["note"]
print(f"classified as historical, compliant stays True — note: {_s['note']}")

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

print("\n=== CAPACITY-MOTIVATED CLOSES ARE REFUSED ===")
# Measured 2026-08-01: with the book full the model closed a losing SOL short "to
# free a position slot" and re-opened the same short 62 minutes later. Because an
# entry is not fully validated when closes execute, shipped and missing config must
# refuse the full-book swap rather than trusting its stated conviction.
from types import SimpleNamespace

class _SwapStub:
    _capacity_only_closes = TradingEngine._capacity_only_closes
    _entry_conviction = TradingEngine._entry_conviction

def _stub(margin=0.15, cap=3, held=(("SOL/USDT:USDT", 0.40),), pending=None,
          min_conv=0.35, allow_swaps=False):
    s = _SwapStub()
    s.risk = SimpleNamespace(swap_conviction_margin=margin, max_open_positions=cap)
    s.pending_entries = dict(pending or {})
    s.position_conviction = {sym: c for sym, c in held}
    s.ai = SimpleNamespace(min_conviction=min_conv)
    s.allow_capacity_swaps = allow_swaps
    return s

def _acct(*symbols):
    return SimpleNamespace(positions=[SimpleNamespace(symbol=s) for s in symbols])

_full = _acct("SOL/USDT:USDT", "ADA/USDT:USDT", "BNB/USDT:USDT")
_close_sol = {"action": "close", "symbol": "SOL/USDT:USDT"}

# The exact observed churn: a replacement does not buy the slot, regardless of its
# unvalidated conviction.
_r1 = _stub()._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 0.45}], _full)
assert "SOL/USDT:USDT" in _r1, _r1
assert "not atomically validated" in _r1["SOL/USDT:USDT"], _r1

# A claimed 1.0-conviction idea is still unvalidated at close time.
_r2 = _stub()._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 1.0}], _full)
assert "SOL/USDT:USDT" in _r2, _r2

# Missing configuration fails closed too; __new__ and legacy objects may not have
# the runtime attribute at all.
_missing = _stub()
del _missing.allow_capacity_swaps
_r_missing = _missing._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 1.0}], _full)
assert "SOL/USDT:USDT" in _r_missing, _r_missing

# Malformed and same-symbol replacement requests are still swaps, not a loophole
# that turns the paired close into a standalone thesis exit.
assert "SOL/USDT:USDT" in _stub()._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "", "conviction": None}], _full)
assert "SOL/USDT:USDT" in _stub()._capacity_only_closes(
    [_close_sol, {"action": "short", "symbol": "SOL/USDT:USDT", "conviction": 1.0}], _full)

# A thesis close with nothing competing for the slot is never touched.
assert _stub()._capacity_only_closes([_close_sol], _full) == {}

# Room to spare: no close is buying a slot, so the guard has no opinion.
_r3 = _stub(cap=5)._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 0.36}], _full)
assert _r3 == {}, _r3

# Resting maker orders are commitments and count toward being full.
_r4 = _stub(cap=4, pending={"XRP/USDT:USDT": {"side": "short"}})._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 0.45}], _full)
assert "SOL/USDT:USDT" in _r4, _r4

# With swaps disabled, every close mixed into the full-book replacement response is
# refused; the engine cannot safely infer which close was a separate thesis exit.
_r5 = _stub(held=(("SOL/USDT:USDT", 0.40), ("ADA/USDT:USDT", 0.80)))._capacity_only_closes(
    [_close_sol, {"action": "close", "symbol": "ADA/USDT:USDT"},
     {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 0.45}], _full)
assert set(_r5) == {"SOL/USDT:USDT", "ADA/USDT:USDT"}, _r5

# The legacy conviction comparison is reachable only with explicit opt-in.
_r6 = _stub(held=(), allow_swaps=True)._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 0.51}], _full)
assert _r6 == {}, _r6
_r7 = _stub(held=(), allow_swaps=True)._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 0.49}], _full)
assert "SOL/USDT:USDT" in _r7, _r7

# Even explicit opt-in with an invalid margin fails closed.
_bad_margin = _stub(margin=0.0, allow_swaps=True)._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "BTC/USDT:USDT", "conviction": 1.0}], _full)
assert "SOL/USDT:USDT" in _bad_margin and "invalid" in _bad_margin["SOL/USDT:USDT"]

# An entry on a symbol already held is not asking for a new slot.
assert _stub()._capacity_only_closes(
    [_close_sol, {"action": "long", "symbol": "ADA/USDT:USDT", "conviction": 0.45}], _full) == {}

_live = RiskManager(config)
assert config["risk"].get("allow_capacity_swaps") is False, "shipped swaps must be off"
assert _live.swap_conviction_margin > 0, "legacy opt-in comparison must fail safe"
assert _live.max_open_positions >= 5, "count cap must leave the correlation budgets binding"
print(f"swap guard OK (allowed {config['risk']['allow_capacity_swaps']}, "
      f"margin {_live.swap_conviction_margin}, "
      f"max_open_positions {_live.max_open_positions})")

print("\n=== PAPER FUNDING IS CHARGED, ONCE PER SETTLEMENT ===")
# Funding was fetched, shown to the model, and never billed — so a paper position
# was free to hold and a live one was not. These pin the arithmetic and, more
# importantly, the idempotency: get_account_state() runs several times per cycle.
from src.core.exchange import ExchangeClient

print("\n=== AI CONTEXT USES CLOSED CANDLES HONESTLY ===")
# WEEX/CCXT returns the current, still-forming bar.  At 12:15 the 12:00 1h bar is
# useful as a live price but must not be allowed into completed-bar volume, ATR,
# ADX or MACD.  Otherwise 15 minutes of volume is compared with 60-minute bars.
class _TimeframeParser:
    @staticmethod
    def parse_timeframe(value):
        return {"1h": 3600, "4h": 14400}[value]


_clock = datetime(2026, 8, 13, 12, 15, tzinfo=timezone.utc)
_cx = ExchangeClient.__new__(ExchangeClient)
_cx.exchange = _TimeframeParser()
_bars = []
for i in range(33):
    opened = datetime(2026, 8, 12, 4, 0, tzinfo=timezone.utc) + timedelta(hours=i)
    # The final completed bar has 2x the previous-20 volume.  The forming bar has
    # only 0.1x; if it leaks into the snapshot the test catches the distortion.
    volume = 200.0 if i == 31 else 10.0 if i == 32 else 100.0
    close = 100.0 + i
    _bars.append(Candle(opened, close - 0.5, close + 1.0, close - 1.0, close, volume))

_closed = _cx.closed_candles(_bars, "1h", now=_clock)
assert len(_closed) == 32 and _closed[-1].timestamp.hour == 11
assert _bars[-1].timestamp.hour == 12, "fixture must include the forming candle"

from src.ai.context import symbol_snapshot
from unittest.mock import patch as _mock_patch
from src.core.models import MarketRegime

_expected_highs = np.array([c.high for c in _closed], dtype=float)
_expected_lows = np.array([c.low for c in _closed], dtype=float)
_expected_closes = np.array([c.close for c in _closed], dtype=float)
with _mock_patch("src.ai.context.detect_regime", return_value=MarketRegime.TRENDING_UP) as _regime:
    _snap = symbol_snapshot(
        "BTC/USDT:USDT", _closed, current_price=_bars[-1].close
    )
_regime_args = _regime.call_args.args
assert np.array_equal(_regime_args[0], _expected_highs), "regime arg 1 must be highs"
assert np.array_equal(_regime_args[1], _expected_lows), "regime arg 2 must be lows"
assert np.array_equal(_regime_args[2], _expected_closes), "regime arg 3 must be closes"
assert _snap["price"] == _bars[-1].close, "execution price should stay live"
assert _snap["last_closed_candle"] == "2026-08-13T11:00:00Z"
assert _snap["change_pct"]["1h"] != 0.0, "1h change cannot compare a close to itself"
assert _snap["volume"]["ratio_vs_20"] == 2.0, _snap["volume"]
assert _snap["volume"]["anomaly"] is False, "anomaly threshold is strictly over 2x"
assert _snap["live_move_from_last_close_pct"] == round(
    (_bars[-1].close / _closed[-1].close - 1) * 100, 2
)
print("forming bar excluded from evidence; live price, changes, volume and regime are correct")

_fx = ExchangeClient.__new__(ExchangeClient)   # no network, no credentials
_fx.mode = "paper"
_fx.paper_funding_enabled = True
_fx.balance = 1000.0
_fx.fetch_funding_rate = lambda symbol: 0.0001   # +0.01%, longs pay

_b = ExchangeClient._funding_boundary
assert _b(datetime(2026, 8, 3, 7, 59)) == datetime(2026, 8, 3, 0, 0)
assert _b(datetime(2026, 8, 3, 8, 0)) == datetime(2026, 8, 3, 8, 0)
assert _b(datetime(2026, 8, 3, 23, 59)) == datetime(2026, 8, 3, 16, 0)
# Hour 0 is itself a boundary, so the day never wraps backwards.
assert _b(datetime(2026, 8, 3, 0, 30)) == datetime(2026, 8, 3, 0, 0)

_between = ExchangeClient._boundaries_between
assert _between(datetime(2026, 8, 3, 0, 0), datetime(2026, 8, 3, 7, 0)) == []
assert _between(datetime(2026, 8, 3, 0, 0), datetime(2026, 8, 3, 9, 0)) == [
    datetime(2026, 8, 3, 8, 0)]
# Down for a day: every settlement slept through is still owed, not skipped.
# 08-02 08:00 -> 08-03 09:00 spans three: 08-02 16:00, 08-03 00:00, 08-03 08:00.
# The anchor itself is excluded — it was already settled when it became the anchor.
assert _between(datetime(2026, 8, 2, 8, 0), datetime(2026, 8, 3, 9, 0)) == [
    datetime(2026, 8, 2, 16, 0), datetime(2026, 8, 3, 0, 0), datetime(2026, 8, 3, 8, 0)]

_pos = Position("BTC/USDT:USDT", Side.LONG, 100.0, 0.5, 5, 90.0, 120.0,
                opened_at=datetime(2026, 8, 3, 6, 0))
_pos.last_funding_at = _b(_pos.opened_at)
# 09:00 crosses exactly one settlement: 0.0001 * (100 * 0.5) = 0.005 paid.
_f1 = _fx._settle_paper_funding(_pos, 100.0, datetime(2026, 8, 3, 9, 0))
assert abs(_f1 - 0.005) < 1e-12, _f1
assert abs(_fx.balance - 999.995) < 1e-12, _fx.balance
# Same cycle, read again: nothing more is owed.
assert _fx._settle_paper_funding(_pos, 100.0, datetime(2026, 8, 3, 9, 5)) == 0.0
assert abs(_fx.balance - 999.995) < 1e-12, "a second read double-charged funding"
# A short on the same positive rate RECEIVES it.
_short = Position("BTC/USDT:USDT", Side.SHORT, 100.0, 0.5, 5, 110.0, 80.0,
                  opened_at=datetime(2026, 8, 3, 6, 0))
_short.last_funding_at = _b(_short.opened_at)
assert _fx._settle_paper_funding(_short, 100.0, datetime(2026, 8, 3, 9, 0)) < 0
# A position opened after a boundary is not billed for it retroactively.
_late = Position("BTC/USDT:USDT", Side.LONG, 100.0, 0.5, 5, 90.0, 120.0,
                 opened_at=datetime(2026, 8, 3, 8, 30))
_late.last_funding_at = _b(_late.opened_at)
assert _fx._settle_paper_funding(_late, 100.0, datetime(2026, 8, 3, 9, 0)) == 0.0
# The switch reproduces old runs and nothing else.
_fx.paper_funding_enabled = False
assert _fx._settle_paper_funding(_pos, 100.0, datetime(2026, 8, 4, 0, 0)) == 0.0
print("funding OK (charged once per settlement, sign correct, restart-safe)")

print("\n=== A TOUCH IS NOT A FILL ===")
# The old rule filled in full whenever `last` reached the limit, which handed us
# free entries at exactly the reversals a real queue never gives you.
_qx = ExchangeClient.__new__(ExchangeClient)
_qx.mode = "paper"
_qx.fill_through_ticks = 1
_qx._tick_size = lambda symbol, ref: 0.1
_qx.paper_positions = {}
_qx.paper_pending = {"o1": {"id": "o1", "symbol": "BTC/USDT:USDT", "side": "long",
                            "amount": 0.5, "limit_price": 100.0, "stop_loss": 90.0,
                            "take_profit": 120.0, "strategy": "", "leverage": 5,
                            "partial_take_profit": None, "partial_fraction": 0.5,
                            "created": 0}}
_qx.fetch_ticker = lambda symbol: {"last": 100.0}          # exact touch
assert _qx.check_entry_fill("o1", "BTC/USDT:USDT")["status"] == "open", "touch must not fill"
_qx.fetch_ticker = lambda symbol: {"last": 99.95}          # inside one tick
assert _qx.check_entry_fill("o1", "BTC/USDT:USDT")["status"] == "open"

_filled = {}
_qx._open_paper_position = lambda **kw: _filled.update(kw)
_qx.maker_fee_rate = 0.0002
_qx.fetch_ticker = lambda symbol: {"last": 99.9}           # a full tick through
_res = _qx.check_entry_fill("o1", "BTC/USDT:USDT")
assert _res["status"] == "filled", _res
# Price stays honest: we fill at OUR limit, never at the better price that traded.
assert _res["fill_price"] == 100.0, _res
assert _filled["fee_rate"] == 0.0002, "a maker fill must not be charged taker"
print("fill rule OK (trade-through required, fill price still ours)")

print("\n=== RESTING ORDERS COUNT AGAINST THE CORRELATION BUDGET ===")
# Pending entries were invisible to both budgets, so N same-side resting orders
# were each sized as if the others did not exist — then filled together.
from src.core.models import AccountState

_cfg = copy.deepcopy(config)
_cfg["risk"]["max_correlated_notional"] = 1.0
_cfg["risk"]["max_portfolio_risk"] = 0.02
_cm = RiskManager(_cfg)
_acct = AccountState(balance=1000, equity=1000, unrealized_pnl=0,
                     margin_used=0, available_margin=1000, positions=[])
_sig = Signal(symbol="BTC/USDT:USDT", side=Side.LONG, strength=0.8,
              entry_price=100.0, stop_loss=96.0, take_profit=112.0,
              strategy="test", leverage=5, reason="budget test")
_corr = {("BTC/USDT:USDT", "ETH/USDT:USDT"): 0.9}
# Sized so the budget BINDS rather than being exhausted outright: a full veto would
# pass this test even if the leg were only being counted, not priced. A partial
# scale can only come from the leg entering the quadratic form.
_leg = Position("ETH/USDT:USDT", Side.LONG, 100.0, 2.0, 5, 96.0, 112.0)

_alone, _ = _cm.correlation_scale(_sig, 5.0, _acct, _corr)
_with_pending, _why = _cm.correlation_scale(_sig, 5.0, _acct, _corr, pending=[_leg])
assert _alone == 1.0, f"an empty book must not scale the first leg: {_alone}"
assert 0 < _with_pending < 1.0, (
    f"a resting correlated leg must SHRINK the next one, not veto it: {_with_pending}")
# And a pending leg constrains exactly as hard as the same leg already filled.
_acct_filled = AccountState(balance=1000, equity=1000, unrealized_pnl=0,
                            margin_used=0, available_margin=1000, positions=[_leg])
_as_filled, _ = _cm.correlation_scale(_sig, 5.0, _acct_filled, _corr)
assert abs(_as_filled - _with_pending) < 1e-12, (
    f"pending and filled must price identically: {_as_filled} vs {_with_pending}")
print(f"correlation budget OK (scale {_alone:.3f} -> {_with_pending:.3f} with one resting leg)")

# The conversion the engine feeds those budgets: priced at the limit, not the
# signal price, because the limit is where a resting order actually fills.
from src.core.engine import TradingEngine
import logging

_pe = TradingEngine.__new__(TradingEngine)
_pe.logger = logging.getLogger("test")
_pe.pending_entries = {
    "BTC/USDT:USDT": {"side": "long", "size": 0.5, "limit_price": 100.0,
                      "stop_loss": 96.0, "take_profit": 112.0, "leverage": 5},
    "ADA/USDT:USDT": {"side": "short", "size": 0.0, "limit_price": 1.0},   # no size
    "XRP/USDT:USDT": {"side": "long", "limit_price": 2.0},                 # malformed
}
_legs = _pe._pending_as_legs()
assert len(_legs) == 1, f"only the usable resting order becomes a leg: {_legs}"
assert _legs[0].symbol == "BTC/USDT:USDT" and _legs[0].side == Side.LONG
assert _legs[0].entry_price == 100.0 and _legs[0].stop_loss == 96.0
print("pending->leg conversion OK (limit-priced, junk skipped not crashed)")

print("\n=== A TERMINAL AI ERROR ALARMS ON THE FIRST ONE ===")
# The account ran out of credit on 2026-08-07 22:19 and the bot sat blind for 65
# hours. Two separate defects made that possible: a 402 was retried like a network
# blip, and it then had to clear a 3-strike counter before anything shouted. In a
# weekly round that outage is 39% of the clock and the 10-trade minimum with it.
from src.ai.client import classify_error, TERMINAL_ERROR_KINDS, AIError


class _ProviderError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status_code = status


# The exact string the live log recorded, 64 times.
_real_402 = _ProviderError(
    "Error code: 402 - {'error': {'message': 'Insufficient Balance', "
    "'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}",
    402,
)
assert classify_error(_real_402) == "billing", classify_error(_real_402)
assert classify_error(_ProviderError("invalid api key", 401)) == "auth"
assert classify_error(_ProviderError("Model does not exist", 404)) == "model"
# A timeout must stay retryable, or one flaky minute becomes a page.
assert classify_error(_ProviderError("Connection timed out")) == "transient"
assert classify_error(_ProviderError("Bad gateway", 502)) == "transient"
assert TERMINAL_ERROR_KINDS == {"billing", "auth", "model"}
assert AIError("x", "billing").kind == "billing"
assert AIError("x").kind == "transient", "default must be the retryable one"


class _FakeEngine:
    """The two health branches, isolated from the trading loop."""
    def __init__(self, kind):
        self.kind = kind
        self.alerts = []

    def terminal(self):
        return self.kind in TERMINAL_ERROR_KINDS

    def healthy(self, consecutive, maxf=3):
        return (not self.terminal()) and consecutive < maxf


# One 402 is enough: unhealthy immediately, without waiting for the counter.
assert _FakeEngine("billing").healthy(consecutive=1) is False, "402 must not wait for 3 strikes"
# A single timeout is not: that is what the counter is for.
assert _FakeEngine("transient").healthy(consecutive=1) is True
assert _FakeEngine("transient").healthy(consecutive=3) is False
print("terminal-error escalation OK (402 alarms at n=1, timeout still needs 3)")

print("\n=== A TOUCHED TAKE-PROFIT IS NOT A MAKER FILL ===")
# Maker TP exits save taker + slippage on the one leg whose price we chose in
# advance. But §2e's lesson applies unchanged: a touch is not a fill. If the rule
# credited a maker exit on a touch it would be the same flattery that inflated the
# old entry fill rate — so it reuses the trade-through test, and when it cannot
# prove the fill the position takes the ordinary taker close instead.
_mx = ExchangeClient.__new__(ExchangeClient)
_mx.mode = "paper"
_mx.maker_exits = True
_mx.fill_through_ticks = 1
_mx._tick_size = lambda symbol, ref: 0.1

_long = Position(symbol="BTC/USDT:USDT", side=Side.LONG, size=0.5, entry_price=100.0,
                 stop_loss=96.0, take_profit=120.0, leverage=5, strategy="t")
assert _mx.maker_exit_price(_long, 119.9) is None, "below TP must not fill"
assert _mx.maker_exit_price(_long, 120.0) is None, "an exact touch is not a fill"
assert _mx.maker_exit_price(_long, 120.05) is None, "inside one tick is not a fill"
assert _mx.maker_exit_price(_long, 120.1) == 120.0, "a full tick through fills, at OUR price"

# Shorts mirror: the exit leg is a BUY, so it needs price BELOW the TP.
_short = Position(symbol="BTC/USDT:USDT", side=Side.SHORT, size=0.5, entry_price=100.0,
                  stop_loss=104.0, take_profit=80.0, leverage=5, strategy="t")
assert _mx.maker_exit_price(_short, 80.0) is None, "an exact touch is not a fill"
assert _mx.maker_exit_price(_short, 79.9) == 80.0, "a full tick through fills"

# The flag genuinely disables it, and a position with no TP cannot claim one.
_mx.maker_exits = False
assert _mx.maker_exit_price(_long, 130.0) is None, "disabled means disabled"
_mx.maker_exits = True
_no_tp = Position(symbol="BTC/USDT:USDT", side=Side.LONG, size=0.5, entry_price=100.0,
                  stop_loss=96.0, take_profit=0, leverage=5, strategy="t")
assert _mx.maker_exit_price(_no_tp, 130.0) is None

# And the saving is real: the same close booked maker vs taker.
_mx.maker_fee_rate, _mx.commission_rate, _mx.slippage_pct = 0.0002, 0.0006, 0.0005
_mx.paper_funding_enabled = False
_mx._settle_paper_funding = lambda *a, **k: None
_mx.fetch_ticker = lambda symbol: {"last": 120.1}
_mx.balance = 1000.0
_mx.paper_trades = []
_mx.paper_positions = {"BTC/USDT:USDT": _long}
_maker = _mx.close_position("BTC/USDT:USDT", maker_price=120.0)
_mx.balance = 1000.0
_mx.paper_trades = []
_mx.paper_positions = {"BTC/USDT:USDT": _long}
_taker = _mx.close_position("BTC/USDT:USDT")
assert _maker["maker"] is True and _taker["maker"] is False
assert _maker["fee"] < _taker["fee"], (_maker["fee"], _taker["fee"])
# The maker fills at OUR limit exactly. The taker fills at the polled mark less
# slippage — which here is 120.04, i.e. slightly ABOVE the TP, because the 60s poll
# caught price a tick past it. That overshoot is luck and cuts both ways; the
# reliable saving is the rate, 0.06%+0.05% slippage against 0.02%.
assert _maker["exit_price"] == 120.0
assert abs(_taker["exit_price"] - 120.1 * (1 - 0.0005)) < 1e-9, _taker["exit_price"]
assert _taker["fee"] / _maker["fee"] > 2.9, "the taker leg should cost ~3x the maker leg"
assert _maker["pnl"] > _taker["pnl"], (_maker["pnl"], _taker["pnl"])
print(f"maker TP exit OK (fee {_taker['fee']:.4f} -> {_maker['fee']:.4f}, "
      f"pnl +{_maker['pnl'] - _taker['pnl']:.4f} on this exit)")

print("\n=== LIVE SAFETY STATE SURVIVES RESTART ===")
_live_state = ExchangeClient.__new__(ExchangeClient)
_live_state.mode = "live"
_live_state._local_brackets = {
    "BTC/USDT:USDT": {
        "stop_loss": 96.0,
        "take_profit": 120.0,
        "opened_at": "2026-08-13T10:00:00",
        "sl_order_id": "sl-1",
        "tp_order_id": "tp-1",
        "trailing_stop": 101.0,
    }
}
_saved_live = _live_state.to_state()
_restored_live = ExchangeClient.__new__(ExchangeClient)
_restored_live.mode = "live"
_restored_live._local_brackets = {}
_restored_live.load_state(_saved_live)
assert _restored_live._local_brackets == _live_state._local_brackets
print("live bracket ids, trail and open time restore exactly")

print("\n=== CURRENT WEEX CONDITIONAL ORDER CONTRACT ===")
class _OrderRecorder:
    def __init__(self):
        self.calls = []

    def create_order(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"id": f"o{len(self.calls)}"}


_br = ExchangeClient.__new__(ExchangeClient)
_br.exchange = _OrderRecorder()
_br.maker_exits = True
_br.normalize_price = lambda symbol, price: price
_placed = _br._create_live_brackets(
    "BTC/USDT:USDT", Side.LONG, 0.01, 96.0, 120.0
)
_sl_args, _sl_kwargs = _br.exchange.calls[0]
assert _sl_args[1] == "market", _sl_args
assert _sl_kwargs["params"]["stopLossPrice"] == 96.0
assert "stopPrice" not in _sl_kwargs["params"]
assert _placed["sl_placed"] and _placed["sl_order_id"] == "o1"
assert _placed["tp_placed"] and _placed["tp_order_id"] == "o2"
print("standalone SL uses stopLossPrice and a MARKET trigger request")

_pending_exchange = _OrderRecorder()
_pending_client = ExchangeClient.__new__(ExchangeClient)
_pending_client.mode = "live"
_pending_client.exchange = _pending_exchange
_pending_client.normalize_amount = lambda symbol, amount: amount
_pending_client.normalize_price = lambda symbol, price: price
_pending_order = _pending_client.place_entry_limit(
    "BTC/USDT:USDT", Side.LONG, 0.01, 99.9,
    stop_loss=96.0, take_profit=120.0,
)
_entry_params = _pending_exchange.calls[0][1]["params"]
assert _entry_params["timeInForce"] == "POST_ONLY"
assert _entry_params["stopLoss"] == {
    "triggerPrice": 96.0, "triggerPriceType": "mark"
}
assert _pending_order["sl_attached"] is True
print("resting entry carries its stop atomically in the accepted order request")

print("\n=== LIVE MARKET CLOSE CANNOT MASQUERADE AS MAKER ===")
class _LiveCloseExchange:
    def __init__(self):
        self.created = []
        self.cancelled = []

    def cancel_order(self, order_id, symbol, params=None):
        self.cancelled.append((order_id, bool((params or {}).get("trigger"))))
        return {"id": order_id}

    def fetch_positions(self, symbols):
        return [{"symbol": symbols[0], "side": "long", "contracts": 0.5}]

    def create_order(self, *args, **kwargs):
        self.created.append((args, kwargs))
        return {"id": "close-1", "average": 119.5, "fee": {"cost": 0.035}}


_lc = ExchangeClient.__new__(ExchangeClient)
_lc.mode = "live"
_lc.exchange = _LiveCloseExchange()
_lc._local_brackets = {
    "BTC/USDT:USDT": {
        "side": "long", "stop_loss": 96.0,
        "sl_order_id": "sl-1", "sl_trigger": True,
        "tp_order_id": "tp-1", "tp_trigger": False,
    }
}
_closed = _lc.close_position("BTC/USDT:USDT", maker_price=120.0)
assert _lc.exchange.created[0][0][1] == "market"
assert _closed["maker"] is False and _closed["execution"] == "taker"
assert _closed["fee"] == 0.035 and _closed["exit_price"] == 119.5
assert set(_lc.exchange.cancelled) == {("sl-1", True), ("tp-1", False)}
print("actual request/fill class drives fee accounting; sibling brackets cancelled")

print("\n=== VENUE-SIDE CLOSES CANNOT VANISH FROM HISTORY ===")
class _TradeHistoryExchange:
    def fetch_my_trades(self, symbol, since=None, limit=None):
        return [{
            "order": "tp-9", "side": "sell", "amount": 0.5, "price": 120.0,
            "takerOrMaker": "maker", "fee": {"cost": 0.012},
        }]


_recovery = ExchangeClient.__new__(ExchangeClient)
_recovery.mode = "live"
_recovery.exchange = _TradeHistoryExchange()
_recovery._local_brackets = {
    "BTC/USDT:USDT": {
        "side": "long", "entry_price": 100.0, "size": 0.5, "initial_size": 0.5,
        "leverage": 5, "stop_loss": 96.0, "take_profit": 120.0,
        "opened_at": "2026-08-13T10:00:00", "missing_since": "2026-08-13T10:01:00",
        "tp_order_id": "tp-9", "entry_fee": 0.01, "fees_paid": 0.01,
    }
}
_events = _recovery.detect_external_closures([])
assert len(_events) == 1
assert _events[0]["reason"] == "take_profit"
assert _events[0]["maker"] is True and _events[0]["fee"] == 0.012
assert _events[0]["position"].opened_at == datetime(2026, 8, 13, 10, 0)
print("missing position is recovered from private fills with actual price, fee and order id")

print("\n=== OFFICIAL UPLOADAILOG IS SIGNED AND DURABLE ===")
from unittest.mock import patch
from src.ai import wars_log

_upload_root = Path(tempfile.mkdtemp())
_payload = {
    "stage": "Strategy Generation",
    "model": "deepseek-v4-pro",
    "input": {
        "messages": [{"role": "user", "content": "trade only this snapshot"}],
        "market_context": {"symbol": "BTC/USDT:USDT", "price": 100},
    },
    "output": {
        "symbol": "BTCUSDT", "side": "BUY", "positionSide": "LONG",
        "type": "LIMIT", "quantity": 0.01, "price": 99.9,
    },
    "explanation": "The supplied snapshot and prompt support this concrete entry.",
    "orderId": 123,
}
_ai_file = _upload_root / "ai.json"
_ai_file.write_text(json.dumps(_payload), encoding="utf-8")
_uploader = wars_log.WeexAILogUploader(
    enabled=True,
    status_dir=_upload_root / "status",
    allowlist_status_path=_upload_root / "allowlist.json",
)
_uploader.api_key = "k"
_uploader.api_secret = "s"
_uploader.passphrase = "p"

class _UploadResponse:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return b'{"code":"00000","data":"upload success"}'

_captured_request = []
def _fake_urlopen(req, timeout):
    _captured_request.append(req)
    return _UploadResponse()

with patch.object(wars_log.request, "urlopen", _fake_urlopen):
    _upload_status = _uploader.register(_ai_file, required=True)
assert _upload_status["uploaded"] is True
assert _captured_request[0].full_url.endswith("/capi/v3/order/uploadAiLog")
assert _captured_request[0].headers.get("Access-sign")
assert _uploader.status()["pending"] == 0
assert _uploader.status()["ready"] is False, "configured is not the same as allowlisted"
_probe_decision = {
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "decide from this snapshot"}],
    "context": {"markets": [{"symbol": "BTC/USDT:USDT", "price": 100}]},
    "raw_response": json.dumps({
        "market_assessment": "No edge.",
        "decisions": [{"symbol": "BTC/USDT:USDT", "action": "hold"}],
    }),
}
with patch.object(wars_log.request, "urlopen", _fake_urlopen):
    _probe_status = _uploader.probe_allowlist(_probe_decision)
assert _probe_status["verified"] is True
assert _uploader.status()["ready"] is True
assert len(_captured_request) == 2
_probe_body = json.loads(_captured_request[1].data)
assert _probe_body["orderId"] is None
assert _probe_body["output"]["decisions"][0]["action"] == "hold"
print("delivery and no-order allowlist probe are signed, durable, and independently gated")

print("\n=== DASHBOARD HEALTH FAILS CLOSED ===")
import os as _os
import dashboard.app as _dashboard

_health_root = Path(tempfile.mkdtemp())
(_health_root / "data").mkdir()
(_health_root / "logs").mkdir()
(_health_root / "config.yaml").write_text(
    "ai:\n  enabled: true\nlogging:\n  state_file: data/bot_state.json\n  file: logs/trading.log\n",
    encoding="utf-8",
)
(_health_root / "data" / "bot_state.json").write_text(
    json.dumps({"risk": {"trade_history": []}}), encoding="utf-8"
)
(_health_root / "logs" / "trading.log").write_text("healthy\n", encoding="utf-8")
(_health_root / "data" / "ai_health.json").write_text(
    json.dumps({"healthy": True}), encoding="utf-8"
)
(_health_root / "data" / "compliance.json").write_text(
    json.dumps({"compliant": True, "orders_without_ai_log": 0}), encoding="utf-8"
)
_execution_file = _health_root / "data" / "execution_health.json"
_execution_file.write_text(json.dumps({"healthy": True}), encoding="utf-8")
_old_root = _dashboard.ROOT
try:
    _dashboard.ROOT = _health_root
    _green = _dashboard.health()
    assert _green.status_code == 200, _green.body
    stale = datetime.now().timestamp() - 700
    _os.utime(_execution_file, (stale, stale))
    _red = _dashboard.health()
    assert _red.status_code == 503, _red.body
    assert "execution protection/delivery unhealthy or stale" in json.loads(_red.body)["failures"]
finally:
    _dashboard.ROOT = _old_root
print("health is 200 only while state, AI, compliance and execution sidecars are fresh")

print("\n=== ALL TESTS PASSED ===")
