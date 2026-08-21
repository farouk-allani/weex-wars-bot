"""Focused regressions for AI context, pace and paper execution safety."""

from __future__ import annotations

import copy
import json
import logging
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.ai.context import build_context
from src.ai.trader import AITrader
from src.core.engine import TradingEngine
from src.core.exchange import ExchangeClient
from src.core.models import AccountState, Candle, Position, Side, TradeResult
from src.risk.manager import RiskManager


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class AIContextTests(unittest.TestCase):
    def test_long_only_and_auditable_rationale_are_enforced(self):
        config = copy.deepcopy(load_config())
        config["competition"]["long_only"] = True
        trader = AITrader(config, client=None, logbook=None)
        decision = {
            "symbol": "BTC/USDT:USDT",
            "action": "short",
            "conviction": 0.7,
            "stop_loss": 102.0,
            "take_profit": 96.0,
            "rationale": "bearish break with failed retest; invalid above 102",
        }
        signal, reason = trader.to_signal(
            decision, "BTC/USDT:USDT", 100.0, 1.0, {"BTC/USDT:USDT"}
        )
        self.assertIsNone(signal)
        self.assertIn("long_only", reason)

        decision.update(action="long", stop_loss=98.0, take_profit=104.0, rationale="")
        signal, reason = trader.to_signal(
            decision, "BTC/USDT:USDT", 100.0, 1.0, {"BTC/USDT:USDT"}
        )
        self.assertIsNone(signal)
        self.assertIn("missing rationale", reason)

    def test_context_exposes_entry_thesis_and_actual_hard_limits(self):
        config = load_config()
        risk = RiskManager(config)
        position = Position(
            "BTC/USDT:USDT", Side.LONG, 100.0, 1.0, 5, 96.0, 108.0
        )
        account = AccountState(1000, 1000, 0, 20, 980, [position])
        thesis = {
            "rationale": "breakout with 4h alignment; invalid below prior swing",
            "entry_market": {"trend": {"regime": "trending_up"}},
        }
        context = build_context(
            symbols_data=[],
            account=account,
            risk=risk,
            recent_trades=[],
            position_theses={position.symbol: thesis},
            trading_constraints={
                "min_rr": 1.35,
                "min_stop_atr": 2.0,
                "max_stop_atr": 4.0,
                "allow_capacity_swaps": False,
            },
        )
        self.assertEqual(context["open_positions"][0]["entry_thesis"], thesis)
        limits = context["hard_limits"]
        self.assertEqual(limits["min_reward_risk"], 1.35)
        self.assertEqual(limits["min_stop_atr"], 2.0)
        self.assertEqual(limits["max_stop_atr"], 4.0)
        self.assertIs(limits["capacity_swaps_allowed"], False)

    def test_parse_failure_is_written_as_a_log_error(self):
        class Client:
            model = "test-model"

            @staticmethod
            def decide(system, user):
                return {
                    "content": "not-json",
                    "model": "test-model",
                    "usage": {"completion_tokens": 8000},
                    "latency_ms": 1,
                }

            @staticmethod
            def parse_json(content):
                raise ValueError("truncated response")

        class Log:
            captured = None

            def record(self, **kwargs):
                self.captured = kwargs
                return "decision-id"

        log = Log()
        trader = AITrader(load_config(), Client(), log)
        decisions, _, _ = trader.decide({"markets": []})
        self.assertEqual(decisions, [])
        self.assertIn("parse error", trader.last_error)
        self.assertEqual(log.captured["error"], trader.last_error)

    def test_position_thesis_is_recovered_from_original_decision(self):
        symbol = "BTC/USDT:USDT"
        record = {
            "timestamp": "2026-08-20T10:00:00Z",
            "raw_response": json.dumps({"market_assessment": "broad trend is up"}),
            "decisions": [{
                "symbol": symbol,
                "action": "long",
                "stop_loss": 96,
                "take_profit": 108,
                "rationale": "clean breakout; invalid below the prior 4h swing",
            }],
            "context": {"markets": [{
                "symbol": symbol,
                "price": 100,
                "trend": {"regime": "trending_up", "htf_4h_direction": "up"},
            }]},
        }
        engine = TradingEngine.__new__(TradingEngine)
        engine.decision_log = SimpleNamespace(get_decision=lambda decision_id: record)
        engine.position_decisions = {symbol: "decision-id"}
        account = SimpleNamespace(
            positions=[Position(symbol, Side.LONG, 100, 1, 5, 96, 108)]
        )
        recovered = engine._position_theses(account)[symbol]
        self.assertIn("clean breakout", recovered["rationale"])
        self.assertEqual(recovered["entry_market"]["trend"]["regime"], "trending_up")
        self.assertEqual(recovered["market_assessment"], "broad trend is up")


class PaceAndGateTests(unittest.TestCase):
    def test_pace_uses_entry_time_and_counts_open_fill(self):
        now = datetime.utcnow()
        closed = TradeResult(
            "BTC/USDT:USDT", Side.LONG, 100, 101, 1, 5, 1, 5,
            2 * 86400, "take_profit", "ai_deepseek", timestamp=now,
        )
        open_position = Position(
            "ETH/USDT:USDT", Side.LONG, 100, 1, 5, 96, 108,
            opened_at=now - timedelta(hours=3), strategy="ai_deepseek",
        )
        engine = TradingEngine.__new__(TradingEngine)
        engine.risk = SimpleNamespace(is_keepalive=lambda strategy: False)
        events = engine._entry_events(SimpleNamespace(positions=[open_position]), [closed])
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual((now - min(events)).total_seconds(), 2 * 86400, delta=1)
        pace = TradingEngine._trade_pace(None, {"min_trades": 10, "round_days": 7}, events)
        self.assertEqual(pace["trades_last_7d"], 2)

    def test_post_close_gate_cannot_bypass_live_protection(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.config = {"competition": {"rules_verified": True}}
        engine.risk = SimpleNamespace(
            can_trade=lambda account: (
                (len(account.positions) < 1),
                "OK" if len(account.positions) < 1 else "Max positions: 1/1",
            )
        )
        engine.exchange = SimpleNamespace(
            mode="live",
            protection_status=lambda: {"healthy": False, "unprotected": ["BTC"]},
        )
        engine.decision_log = None
        full = SimpleNamespace(positions=[object()])
        freed = SimpleNamespace(positions=[])
        self.assertFalse(engine._entry_gate(full)[0])
        allowed, reason = engine._entry_gate(freed)
        self.assertFalse(allowed)
        self.assertIn("execution protection unresolved", reason)

        engine.config["competition"]["rules_verified"] = False
        allowed, reason = engine._entry_gate(freed)
        self.assertFalse(allowed)
        self.assertIn("rules have not been verified", reason)

    def test_round_start_accepts_timezone_aware_iso(self):
        started = datetime.now(timezone.utc) - timedelta(days=1)
        event = datetime.utcnow() - timedelta(hours=2)
        pace = TradingEngine._trade_pace(
            None,
            {
                "min_trades": 10,
                "round_days": 7,
                "round_started": started.isoformat(),
            },
            [event],
        )
        self.assertTrue(pace["round_anchored"])
        self.assertEqual(pace["trades_this_round"], 1)

    def test_pending_orders_refresh_gate_between_attempts(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.pending_entries = {"A": {}, "B": {}}
        engine.exchange = SimpleNamespace(get_account_state=lambda: object())
        engine.logger = logging.getLogger("pending-gate-test")
        gate_calls = []
        managed = []

        def gate(_account):
            gate_calls.append(1)
            return (len(gate_calls) == 1, "blocked after first fill")

        engine._entry_gate = gate
        engine._manage_one_pending = lambda symbol, pe: managed.append(symbol)
        engine._abandon_pending = (
            lambda symbol, pe, reason: managed.append(f"cancel:{symbol}")
        )
        engine._manage_pending_entries()
        self.assertEqual(len(gate_calls), 2)
        self.assertEqual(managed, ["A", "cancel:B"])

    def test_rules_gate_is_rechecked_after_position_management(self):
        engine = TradingEngine.__new__(TradingEngine)
        account = SimpleNamespace(positions=[])
        engine.exchange = SimpleNamespace(get_account_state=lambda: account)
        engine.strategy = SimpleNamespace(sync_scores_from_risk=lambda risk: None)
        engine.risk = object()
        engine.pending_entries = {}
        engine.logger = logging.getLogger("rules-gate-test")
        engine._reconcile_external_closures = lambda current: None
        engine._manage_positions = lambda current: None
        answers = iter([(True, "OK"), (False, "loss cooldown")])
        gate_calls = []

        def gate(_account):
            gate_calls.append(1)
            return next(answers)

        engine._entry_gate = gate
        engine._run_rules_cycle([], "1h", "4h", 100, 80)
        self.assertEqual(len(gate_calls), 2)


class PaperExecutionTests(unittest.TestCase):
    def test_completed_bar_brackets_use_conservative_intrabar_order(self):
        position = Position("BTC/USDT:USDT", Side.LONG, 100, 1, 5, 95, 105)
        ambiguous = Candle(datetime.utcnow(), 100, 106, 94, 101, 1)
        self.assertEqual(
            TradingEngine._paper_bar_hit(position, ambiguous),
            ("stop_loss", 95.0, False),
        )
        gap_target = Candle(datetime.utcnow(), 106, 108, 94, 100, 1)
        self.assertEqual(
            TradingEngine._paper_bar_hit(position, gap_target),
            ("take_profit", 105.0, True),
        )
        gap_stop = Candle(datetime.utcnow(), 93, 96, 92, 94, 1)
        self.assertEqual(
            TradingEngine._paper_bar_hit(position, gap_stop),
            ("stop_loss", 93.0, False),
        )
        touch_only = Candle(datetime.utcnow(), 100, 105, 99, 104, 1)
        self.assertIsNone(TradingEngine._paper_bar_hit(position, touch_only, 0.1))
        traded_through = Candle(datetime.utcnow(), 100, 105.1, 99, 104, 1)
        self.assertEqual(
            TradingEngine._paper_bar_hit(position, traded_through, 0.1),
            ("take_profit", 105.0, True),
        )

    def test_completed_bar_is_not_processed_twice(self):
        opened = datetime(2026, 8, 20, 12, 0, 30)
        bars = [
            Candle(datetime(2026, 8, 20, 12, 0), 100, 101, 99, 100, 1),
            Candle(datetime(2026, 8, 20, 12, 1), 100, 101, 94, 96, 1),
        ]
        engine = TradingEngine.__new__(TradingEngine)
        engine.paper_bar_checked = {}
        engine.exchange = SimpleNamespace(
            fetch_candles=lambda *args: bars,
            closed_candles=lambda raw, timeframe: raw,
        )
        position = Position(
            "BTC/USDT:USDT", Side.LONG, 100, 1, 5, 95, 105, opened_at=opened
        )
        self.assertEqual(
            engine._paper_completed_bar_exit(position),
            ("stop_loss", 95.0, False),
        )
        self.assertIsNone(engine._paper_completed_bar_exit(position))

    def test_entry_minute_bar_is_processed_once(self):
        opened = datetime(2026, 8, 20, 12, 0, 30)
        entry_bar = Candle(
            datetime(2026, 8, 20, 12, 0), 100, 101, 94, 96, 1
        )
        engine = TradingEngine.__new__(TradingEngine)
        engine.paper_bar_checked = {}
        engine.exchange = SimpleNamespace(
            maker_exits=False,
            fetch_candles=lambda *args: [entry_bar],
            closed_candles=lambda raw, timeframe: raw,
        )
        position = Position(
            "BTC/USDT:USDT", Side.LONG, 100, 1, 5, 95, 105,
            opened_at=opened,
        )
        self.assertEqual(
            engine._paper_completed_bar_exit(position),
            ("stop_loss", 95.0, False),
        )
        self.assertIsNone(engine._paper_completed_bar_exit(position))

    def test_entry_minute_target_is_ignored_but_later_target_counts(self):
        opened = datetime(2026, 8, 20, 12, 0, 30)
        bars = [Candle(datetime(2026, 8, 20, 12, 0), 100, 106, 99, 105, 1)]
        engine = TradingEngine.__new__(TradingEngine)
        engine.paper_bar_checked = {}
        engine.exchange = SimpleNamespace(
            maker_exits=False,
            fetch_candles=lambda *args: list(bars),
            closed_candles=lambda raw, timeframe: raw,
        )
        position = Position(
            "BTC/USDT:USDT", Side.LONG, 100, 1, 5, 95, 105,
            opened_at=opened,
        )
        self.assertIsNone(engine._paper_completed_bar_exit(position))

        bars.append(Candle(datetime(2026, 8, 20, 12, 1), 104, 106, 103, 105, 1))
        self.assertEqual(
            engine._paper_completed_bar_exit(position),
            ("take_profit", 105.0, False),
        )
        self.assertIsNone(engine._paper_completed_bar_exit(position))

    def test_initial_maker_size_uses_normalized_bracket_geometry(self):
        placed = {}

        class Exchange:
            mode = "paper"

            @staticmethod
            def touch_price(symbol, side):
                return 100.0

            @staticmethod
            def get_account_state():
                return SimpleNamespace(equity=1000.0)

            @staticmethod
            def normalize_price(symbol, price):
                return 89.0 if price == 90.0 else price

            @staticmethod
            def normalize_amount(symbol, amount):
                return amount

            @staticmethod
            def place_entry_limit(symbol, side, size, price, **kwargs):
                placed.update(size=size, price=price, **kwargs)
                return {
                    "id": "normalized-order",
                    "amount": size,
                    "limit_price": price,
                    "stop_loss": kwargs["stop_loss"],
                    "take_profit": kwargs["take_profit"],
                    "sl_attached": True,
                }

        engine = TradingEngine.__new__(TradingEngine)
        engine.exchange = Exchange()
        engine.risk = SimpleNamespace(max_risk_per_trade=0.02, min_position_usd=0)
        engine.logger = logging.getLogger("initial-maker-normalization-test")
        engine.pending_entries = {}
        engine.ai = None
        engine.decision_log = None
        engine.config = {"competition": {"min_rr": 1.35}}
        engine.max_chase_atr = 0.5
        engine.entry_ttl_sec = 2700
        engine._persist_state = lambda: None
        signal = SimpleNamespace(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry_price=100.0,
            stop_loss=90.0, take_profit=120.0, partial_take_profit=None,
            partial_fraction=0.5, strategy="ai_deepseek", leverage=5,
            strength=0.5,
        )
        engine._place_maker_entry(signal, 1.0, atr=5.0)
        self.assertAlmostEqual(placed["size"], 10.0 / 11.0)
        self.assertLessEqual(placed["size"] * (100.0 - 89.0), 10.0 + 1e-9)

    def test_unsafe_initial_maker_response_is_cancelled(self):
        cancelled = []

        class Exchange:
            mode = "paper"

            @staticmethod
            def touch_price(symbol, side):
                return 100.0

            @staticmethod
            def get_account_state():
                return SimpleNamespace(equity=1000.0)

            @staticmethod
            def place_entry_limit(symbol, side, size, price, **kwargs):
                return {
                    "id": "unsafe-order", "amount": size,
                    "limit_price": price, "stop_loss": 89.0,
                    "take_profit": kwargs["take_profit"], "sl_attached": True,
                }

            @staticmethod
            def cancel_entry(order_id, symbol):
                cancelled.append(order_id)
                return {"cancelled": True, "filled_amount": 0}

        engine = TradingEngine.__new__(TradingEngine)
        engine.exchange = Exchange()
        engine.risk = SimpleNamespace(max_risk_per_trade=0.02, min_position_usd=0)
        engine.logger = logging.getLogger("unsafe-initial-maker-test")
        engine.pending_entries = {}
        engine.ai = None
        engine.decision_log = None
        engine.config = {"competition": {"min_rr": 1.35}}
        engine.max_chase_atr = 0.5
        engine.entry_ttl_sec = 2700
        engine._persist_state = lambda: None
        signal = SimpleNamespace(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry_price=100.0,
            stop_loss=90.0, take_profit=120.0, partial_take_profit=None,
            partial_fraction=0.5, strategy="ai_deepseek", leverage=5,
            strength=0.5,
        )
        engine._place_maker_entry(signal, 1.0, atr=5.0)
        self.assertEqual(cancelled, ["unsafe-order"])
        self.assertEqual(engine.pending_entries, {})

    def test_unsafe_response_cancel_race_brackets_then_flattens_fill(self):
        symbol = "BTC/USDT:USDT"
        position = SimpleNamespace(symbol=symbol)
        bracketed = []
        closed = []

        engine = TradingEngine.__new__(TradingEngine)
        engine.exchange = SimpleNamespace(
            cancel_entry=lambda order_id, order_symbol: {
                "cancelled": False, "filled_amount": 0.25,
            },
            get_account_state=lambda: SimpleNamespace(positions=[position]),
        )
        engine.logger = logging.getLogger("unsafe-cancel-race-test")
        engine._on_entry_fill = lambda *args: bracketed.append(args)
        engine._close_position = lambda *args: closed.append(args)
        pending = {"order_id": "unsafe-order", "limit_price": 100.0}

        engine._abandon_pending(
            symbol, pending, "unsafe_order_response:stop_dollar_budget"
        )

        self.assertEqual(bracketed[0][0:3], (symbol, pending, 0.25))
        self.assertEqual(closed, [(position, 100.0, "risk_budget_violation")])

    def test_maker_reprice_shrinks_size_to_original_stop_budget(self):
        placed = {}

        class Exchange:
            mode = "paper"

            @staticmethod
            def check_entry_fill(order_id, symbol):
                return {"status": "open"}

            @staticmethod
            def touch_price(symbol, side):
                return 105.0

            @staticmethod
            def cancel_entry(order_id, symbol):
                return {"cancelled": True, "filled_amount": 0}

            @staticmethod
            def get_account_state():
                return SimpleNamespace(equity=1000.0)

            @staticmethod
            def place_entry_limit(symbol, side, size, price, **kwargs):
                placed.update(size=size, price=price)
                return {
                    "id": "new-order", "amount": size, "limit_price": price,
                    "stop_loss": kwargs["stop_loss"],
                    "take_profit": kwargs["take_profit"], "sl_attached": True,
                }

        engine = TradingEngine.__new__(TradingEngine)
        engine.exchange = Exchange()
        engine.risk = SimpleNamespace(max_risk_per_trade=0.012, min_position_usd=10)
        engine.entry_ttl_sec = 3600
        engine.max_chase_atr = 1.0
        engine.reprice_sec = 0
        engine.decision_log = None
        engine.logger = logging.getLogger("maker-risk-test")
        engine._persist_state = lambda: None
        old = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
        engine.pending_entries = {"BTC/USDT:USDT": {
            "order_id": "old-order", "side": "long", "size": 1.0,
            "limit_price": 100.0, "signal_price": 100.0,
            "stop_loss": 90.0, "take_profit": 140.0,
            "strategy": "ai_deepseek", "leverage": 5,
            "partial_fraction": 0.5, "atr": 10.0, "min_rr": 1.35,
            "risk_budget_usd": 10.0, "placed_at": old,
            "last_reprice_at": old, "reprices": 0,
        }}
        engine._manage_one_pending("BTC/USDT:USDT", engine.pending_entries["BTC/USDT:USDT"])
        self.assertAlmostEqual(placed["size"], 10.0 / 15.0)
        self.assertLessEqual(placed["size"] * (105.0 - 90.0), 10.0 + 1e-9)

    def test_unsafe_reprice_response_is_cancelled(self):
        cancelled = []

        class Exchange:
            mode = "paper"

            @staticmethod
            def check_entry_fill(order_id, symbol):
                return {"status": "open"}

            @staticmethod
            def touch_price(symbol, side):
                return 105.0

            @staticmethod
            def cancel_entry(order_id, symbol):
                cancelled.append(order_id)
                return {"cancelled": True, "filled_amount": 0}

            @staticmethod
            def get_account_state():
                return SimpleNamespace(equity=1000.0)

            @staticmethod
            def place_entry_limit(symbol, side, size, price, **kwargs):
                return {
                    "id": "unsafe-new-order", "amount": size,
                    "limit_price": price, "stop_loss": 89.0,
                    "take_profit": kwargs["take_profit"], "sl_attached": True,
                }

        engine = TradingEngine.__new__(TradingEngine)
        engine.exchange = Exchange()
        engine.risk = SimpleNamespace(max_risk_per_trade=0.012, min_position_usd=0)
        engine.entry_ttl_sec = 3600
        engine.max_chase_atr = 1.0
        engine.reprice_sec = 0
        engine.decision_log = None
        engine.logger = logging.getLogger("unsafe-reprice-test")
        engine._persist_state = lambda: None
        old = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
        engine.pending_entries = {"BTC/USDT:USDT": {
            "order_id": "old-order", "side": "long", "size": 1.0,
            "limit_price": 100.0, "signal_price": 100.0,
            "stop_loss": 90.0, "take_profit": 140.0,
            "strategy": "ai_deepseek", "leverage": 5,
            "partial_fraction": 0.5, "atr": 10.0, "min_rr": 1.35,
            "risk_budget_usd": 10.0, "placed_at": old,
            "last_reprice_at": old, "reprices": 0,
        }}
        engine._manage_one_pending(
            "BTC/USDT:USDT", engine.pending_entries["BTC/USDT:USDT"]
        )
        self.assertEqual(cancelled, ["old-order", "unsafe-new-order"])
        self.assertEqual(engine.pending_entries, {})

    def test_paper_stop_uses_recovered_trigger_not_later_ticker(self):
        client = ExchangeClient.__new__(ExchangeClient)
        client.mode = "paper"
        client.paper_funding_enabled = False
        client.balance = 1000.0
        client.commission_rate = 0.0
        client.maker_fee_rate = 0.0
        client.paper_trades = []
        client.paper_positions = {
            "BTC/USDT:USDT": Position(
                "BTC/USDT:USDT", Side.LONG, 100, 1, 5, 90, 120
            )
        }
        client.fetch_ticker = lambda symbol: {"last": 110.0}
        client.apply_slippage = lambda price, side, is_exit=False: price
        result = client.close_position("BTC/USDT:USDT", trigger_price=90.0)
        self.assertEqual(result["exit_price"], 90.0)
        self.assertEqual(result["pnl"], -10.0)


class CapacitySwapSafetyTests(unittest.TestCase):
    @staticmethod
    def _engine(*, include_switch: bool = True, allow_swaps: bool = False):
        engine = TradingEngine.__new__(TradingEngine)
        engine.risk = SimpleNamespace(
            max_open_positions=1,
            swap_conviction_margin=0.15,
        )
        engine.pending_entries = {}
        engine.position_conviction = {"SOL/USDT:USDT": 0.40}
        engine.ai = SimpleNamespace(min_conviction=0.35)
        if include_switch:
            engine.allow_capacity_swaps = allow_swaps
        return engine

    @staticmethod
    def _full_account():
        return SimpleNamespace(
            positions=[SimpleNamespace(symbol="SOL/USDT:USDT")]
        )

    @staticmethod
    def _swap_decisions():
        return [
            {"action": "close", "symbol": "SOL/USDT:USDT"},
            {
                "action": "long",
                "symbol": "BTC/USDT:USDT",
                "conviction": 1.0,
            },
        ]

    def test_disabled_capacity_swap_refuses_even_max_conviction_replacement(self):
        blocked = self._engine()._capacity_only_closes(
            self._swap_decisions(), self._full_account()
        )
        self.assertIn("SOL/USDT:USDT", blocked)
        self.assertIn("not atomically validated", blocked["SOL/USDT:USDT"])

    def test_missing_capacity_swap_config_fails_closed(self):
        blocked = self._engine(include_switch=False)._capacity_only_closes(
            self._swap_decisions(), self._full_account()
        )
        self.assertIn("SOL/USDT:USDT", blocked)

    def test_invalid_or_same_symbol_replacement_cannot_hide_the_swap(self):
        engine = self._engine()
        for replacement in (
            {"action": "long", "symbol": "", "conviction": None},
            {"action": "short", "symbol": "SOL/USDT:USDT", "conviction": 1.0},
        ):
            with self.subTest(replacement=replacement):
                blocked = engine._capacity_only_closes(
                    [
                        {"action": "close", "symbol": "SOL/USDT:USDT"},
                        replacement,
                    ],
                    self._full_account(),
                )
                self.assertIn("SOL/USDT:USDT", blocked)

    def test_standalone_thesis_close_remains_allowed(self):
        blocked = self._engine()._capacity_only_closes(
            [{"action": "close", "symbol": "SOL/USDT:USDT"}],
            self._full_account(),
        )
        self.assertEqual(blocked, {})


if __name__ == "__main__":
    unittest.main()
