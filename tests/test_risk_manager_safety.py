from datetime import datetime, time, timedelta
from pathlib import Path
import sys
import unittest

# Keep this focused suite directly runnable without requiring package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import AccountState, Side, Signal, TradeResult
from src.risk.manager import RiskManager


def _config(*, initial_capital=1000.0, daily_loss_limit=0.025):
    return {
        "risk": {
            "max_risk_per_trade": 0.02,
            "max_drawdown": 0.50,
            "max_open_positions": 5,
            "max_consecutive_losses": 3,
            "cooldown_hours": 6,
            "daily_loss_limit": daily_loss_limit,
        },
        "sizing": {
            "method": "fixed_fractional",
            "min_position_usd": 0,
            "max_position_pct": 10.0,
        },
        "backtest": {"initial_capital": initial_capital},
    }


def _account(equity=1000.0, *, balance=None, unrealized_pnl=0.0):
    balance = equity if balance is None else balance
    return AccountState(
        balance=balance,
        equity=equity,
        unrealized_pnl=unrealized_pnl,
        margin_used=0.0,
        available_margin=equity,
        positions=[],
    )


def _signal(strength=0.5):
    return Signal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        strength=strength,
        strategy="adaptive",
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        leverage=5,
        reason="test",
    )


def _trade(pnl):
    return TradeResult(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        entry_price=100.0,
        exit_price=101.0,
        size=1.0,
        leverage=5,
        pnl=pnl,
        pnl_pct=pnl,
        duration_seconds=60,
        exit_reason="test",
        strategy="adaptive",
    )


class RiskManagerSafetyTests(unittest.TestCase):
    def test_trade_policy_cohort_survives_state_round_trip(self):
        manager = RiskManager(_config())
        trade = _trade(1.0)
        trade.policy_id = "paper-v1"
        manager.trade_history = [trade]

        restored = RiskManager(_config())
        restored.load_state(manager.to_state())

        self.assertEqual(restored.trade_history[0].policy_id, "paper-v1")

    def test_adaptive_weights_cannot_raise_stop_risk_above_strength_budget(self):
        manager = RiskManager(_config())
        signal = _signal(strength=0.5)

        manager.get_strategy_weight = lambda _strategy: 1.4
        amount = manager.calculate_position_size(signal, _account(), pair_weight=1.9)
        self.assertAlmostEqual(amount * 10.0, 10.0)

        # A profitable strategy weight cannot undo a pair-level reduction.
        reduced = manager.calculate_position_size(signal, _account(), pair_weight=0.5)
        self.assertAlmostEqual(reduced * 10.0, 5.0)

        # Nor can a strong pair weight undo a strategy-level reduction.
        manager.get_strategy_weight = lambda _strategy: 0.4
        reduced = manager.calculate_position_size(signal, _account(), pair_weight=1.9)
        self.assertAlmostEqual(reduced * 10.0, 4.0)

    def test_winning_trade_does_not_clear_unexpired_loss_cooldown(self):
        manager = RiskManager(_config())
        now = datetime.combine(datetime.utcnow().date(), time(hour=12))
        cooldown_until = now + timedelta(hours=6)
        manager.cooldown_until = cooldown_until

        manager.record_trade(_trade(5.0), now=now)

        self.assertEqual(manager.cooldown_until, cooldown_until)
        allowed, reason = manager.can_trade(_account(), now=now + timedelta(hours=1))
        self.assertFalse(allowed)
        self.assertIn("Loss cooldown", reason)

        allowed, reason = manager.can_trade(
            _account(), now=cooldown_until + timedelta(seconds=1)
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")
        self.assertIsNone(manager.cooldown_until)

    def test_daily_loss_gate_uses_persisted_mark_to_market_equity(self):
        manager = RiskManager(_config(daily_loss_limit=0.025))
        now = datetime.combine(datetime.utcnow().date(), time(hour=12))

        allowed, _ = manager.can_trade(_account(1000.0), now=now)
        self.assertTrue(allowed)
        self.assertAlmostEqual(manager.daily_start_equity, 1000.0)

        # Realized reporting can be positive while unrealized/funding-driven equity
        # is through the loss limit; the gate must follow account equity.
        manager.daily_pnl = 10.0
        allowed, reason = manager.can_trade(
            _account(974.0, balance=990.0, unrealized_pnl=-16.0),
            now=now + timedelta(hours=1),
        )
        self.assertFalse(allowed)
        self.assertIn("2.6%", reason)
        self.assertEqual(manager.daily_pnl, 10.0)

        state = manager.to_state()
        restored = RiskManager(_config(daily_loss_limit=0.025))
        restored.load_state(state)
        self.assertAlmostEqual(restored.daily_start_equity, 1000.0)
        allowed, _ = restored.can_trade(
            _account(974.0), now=now + timedelta(hours=2)
        )
        self.assertFalse(allowed)

        # The first observation on a new UTC day becomes the new baseline.
        allowed, reason = restored.can_trade(
            _account(974.0), now=now + timedelta(days=1)
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")
        self.assertEqual(restored.daily_pnl, 0.0)
        self.assertAlmostEqual(restored.daily_start_equity, 974.0)

    def test_legacy_state_without_daily_equity_baseline_starts_safely(self):
        now = datetime.combine(datetime.utcnow().date(), time(hour=12))
        legacy = RiskManager(_config()).to_state()
        legacy.pop("daily_start_equity")
        legacy["daily_pnl"] = -20.0

        restored = RiskManager(_config(daily_loss_limit=0.025))
        restored.load_state(legacy)
        self.assertIsNone(restored.daily_start_equity)

        allowed, _ = restored.can_trade(_account(980.0), now=now)
        self.assertTrue(allowed)
        self.assertAlmostEqual(restored.daily_start_equity, 980.0)

        allowed, _ = restored.can_trade(
            _account(954.0), now=now + timedelta(hours=1)
        )
        self.assertFalse(allowed)

    def test_get_stats_drawdown_uses_configured_initial_capital(self):
        for initial_capital, expected_drawdown in [(1000.0, 0.10), (2000.0, 0.05)]:
            with self.subTest(initial_capital=initial_capital):
                manager = RiskManager(_config(initial_capital=initial_capital))
                manager.trade_history = [_trade(-100.0)]
                self.assertAlmostEqual(
                    manager.get_stats()["max_drawdown"], expected_drawdown
                )


if __name__ == "__main__":
    unittest.main()
