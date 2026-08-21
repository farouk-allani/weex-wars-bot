import math
import unittest

from check_ready import (
    competition_rules_verified,
    configured_policy_id,
    live_paper_evidence_gates,
    paper_evidence_from_state,
)
from dashboard.app import build_metrics


class DashboardReportingTests(unittest.TestCase):
    def test_complete_history_overrides_capped_rolling_arrays(self):
        state = {
            "risk": {
                "trade_history": [
                    {"strategy": "trend", "symbol": "BTC/USDT:USDT", "pnl": 10},
                    {"strategy": "trend", "symbol": "BTC/USDT:USDT", "pnl": -3},
                    {"strategy": "mean", "symbol": "ETH/USDT:USDT", "pnl": 5},
                ],
                "strategy_pnls": {"trend": [-999]},
                "pair_sharpes": {"BTC/USDT:USDT": [-999]},
            }
        }

        metrics = build_metrics(state, {"backtest": {"initial_capital": 1000}})
        strategies = {row["name"]: row for row in metrics["strategy_stats"]}
        pairs = {row["symbol"]: row for row in metrics["pair_stats"]}

        self.assertEqual(strategies["trend"]["trades"], 2)
        self.assertEqual(strategies["trend"]["pnl"], 7)
        self.assertEqual(strategies["mean"]["pnl"], 5)
        self.assertEqual(pairs["BTC/USDT:USDT"]["trades"], 2)
        self.assertEqual(pairs["BTC/USDT:USDT"]["pnl"], 7)
        self.assertEqual(pairs["BTC/USDT:USDT"]["win_rate"], 0.5)

    def test_empty_current_history_does_not_show_stale_rolling_stats(self):
        state = {
            "risk": {
                "trade_history": [],
                "strategy_pnls": {"stale": [100]},
                "pair_sharpes": {"BTC/USDT:USDT": [100]},
            }
        }

        metrics = build_metrics(state, {})

        self.assertEqual(metrics["strategy_stats"], [])
        self.assertEqual(metrics["pair_stats"], [])

    def test_legacy_state_uses_rolling_arrays_when_history_is_absent(self):
        state = {
            "risk": {
                "strategy_pnls": {"legacy": [2, -1]},
                "pair_sharpes": {"SOL/USDT:USDT": [2, -1]},
            }
        }

        metrics = build_metrics(state, {})

        self.assertEqual(metrics["strategy_stats"][0]["pnl"], 1)
        self.assertEqual(metrics["strategy_stats"][0]["trades"], 2)
        self.assertEqual(metrics["pair_stats"][0]["pnl"], 1)


class SavedPaperEvidenceTests(unittest.TestCase):
    def test_readiness_uses_only_the_declared_policy_cohort(self):
        rows = [
            {"pnl": 2, "policy_id": "current"},
            {"pnl": -1, "policy_id": "current"},
            {"pnl": 999, "policy_id": "old"},
            {"pnl": 999},
        ]

        evidence = paper_evidence_from_state(
            {"risk": {"trade_history": rows}}, policy_id="current"
        )

        self.assertEqual(evidence["closed_trades"], 2)
        self.assertEqual(evidence["profit_factor"], 2)
        self.assertEqual(evidence["other_policy_rows"], 2)
        self.assertEqual(configured_policy_id({"evaluation": {"policy_id": " current "}}), "current")

    def test_competition_rules_require_explicit_true(self):
        self.assertFalse(competition_rules_verified({}))
        self.assertFalse(competition_rules_verified({"competition": {"rules_verified": False}}))
        self.assertFalse(competition_rules_verified({"competition": {"rules_verified": 1}}))
        self.assertTrue(competition_rules_verified({"competition": {"rules_verified": True}}))

    def test_thresholds_pass_on_sufficient_profitable_sample(self):
        pnls = [value for _ in range(10) for value in (2, -1)] + [1] * 20
        state = {"risk": {"trade_history": [{"pnl": pnl} for pnl in pnls]}}

        evidence = paper_evidence_from_state(state)
        gates = live_paper_evidence_gates(evidence)

        self.assertEqual(evidence["closed_trades"], 40)
        self.assertEqual(evidence["profit_factor"], 4)
        self.assertEqual(evidence["recent_net_pnl"], 20)
        self.assertTrue(all(gates.values()))

    def test_empty_sample_has_no_profit_factor_and_fails_all_gates(self):
        evidence = paper_evidence_from_state({})

        self.assertIsNone(evidence["profit_factor"])
        self.assertEqual(evidence["recent_net_pnl"], 0)
        self.assertFalse(any(live_paper_evidence_gates(evidence).values()))

    def test_no_losses_reports_infinite_profit_factor_without_crashing(self):
        state = {"risk": {"trade_history": [{"pnl": 1}] * 40}}

        evidence = paper_evidence_from_state(state)

        self.assertTrue(math.isinf(evidence["profit_factor"]))
        self.assertTrue(all(live_paper_evidence_gates(evidence).values()))

    def test_non_finite_and_malformed_rows_are_not_counted(self):
        rows = [
            {"pnl": "2.5"},
            {"pnl": "nan"},
            {"pnl": float("inf")},
            {"pnl": "bad"},
            {},
            None,
            {"pnl": -1},
        ]

        evidence = paper_evidence_from_state({"risk": {"trade_history": rows}})

        self.assertEqual(evidence["closed_trades"], 2)
        self.assertEqual(evidence["invalid_rows"], 5)
        self.assertEqual(evidence["profit_factor"], 2.5)
        self.assertFalse(live_paper_evidence_gates(evidence)["recent_net_pnl"])


if __name__ == "__main__":
    unittest.main()
