"""Focused regressions for proving maker fills have a live venue stop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.exchange import ExchangeClient
from src.core.models import Side


class Venue:
    def __init__(self, orders=None, *, fetch_error=None, create_error=None):
        self.orders = list(orders or [])
        self.fetch_error = fetch_error
        self.create_error = create_error
        self.fetches = []
        self.created = []

    def fetch_open_orders(self, symbol, params=None):
        self.fetches.append((symbol, params))
        if self.fetch_error:
            raise RuntimeError(self.fetch_error)
        return list(self.orders)

    def create_order(self, *args, **kwargs):
        self.created.append((args, kwargs))
        if self.create_error:
            raise RuntimeError(self.create_error)
        return {"id": "fallback-stop"}


def live_client(venue: Venue) -> ExchangeClient:
    client = ExchangeClient.__new__(ExchangeClient)
    client.mode = "live"
    client.exchange = venue
    client.maker_exits = False
    client.maker_fee_rate = 0.0002
    client.commission_rate = 0.0006
    client.config = {"trading": {"default_leverage": 5}}
    client._local_brackets = {}
    client.normalize_amount = lambda symbol, amount: amount
    client.normalize_price = lambda symbol, price: price
    return client


class DirectEntryVenue(Venue):
    def __init__(self, orders=None, *, fallback_error=None):
        super().__init__(orders)
        self.fallback_error = fallback_error

    def create_order(self, *args, **kwargs):
        self.created.append((args, kwargs))
        if len(self.created) == 1:
            return {
                "id": "market-entry", "average": 100.0,
                "fee": {"cost": 0.015},
            }
        if self.fallback_error:
            raise RuntimeError(self.fallback_error)
        return {"id": "fallback-stop"}


class LiveStopProofTests(unittest.TestCase):
    symbol = "BTC/USDT:USDT"

    def finalize(self, client, *, stop_attached=True):
        return client.finalize_entry_fill(
            self.symbol,
            Side.LONG,
            amount=0.25,
            fill_price=100.0,
            stop_loss=96.0,
            take_profit=0.0,
            stop_attached=stop_attached,
        )

    def test_exact_open_attached_stop_is_confirmed_without_duplicate(self):
        venue = Venue([{
            "id": "attached-stop",
            "symbol": self.symbol,
            "side": "sell",
            "stopPrice": 96.0,
        }])
        client = live_client(venue)

        result = self.finalize(client)

        self.assertEqual(
            venue.fetches,
            [(self.symbol, {"type": "swap", "trigger": True})],
        )
        self.assertEqual(venue.created, [])
        self.assertTrue(result["sl_placed"])
        self.assertEqual(result["sl_order_id"], "attached-stop")
        self.assertEqual(
            client._local_brackets[self.symbol]["sl_order_id"],
            "attached-stop",
        )

    def test_symbol_side_and_trigger_must_all_match_before_avoiding_fallback(self):
        venue = Venue([
            {
                "id": "wrong-symbol", "symbol": "ETH/USDT:USDT",
                "side": "sell", "stopPrice": 96.0,
            },
            {
                "id": "wrong-side", "symbol": self.symbol,
                "side": "buy", "stopPrice": 96.0,
            },
            {
                "id": "wrong-trigger", "symbol": self.symbol,
                "side": "sell", "stopPrice": 95.0,
            },
        ])
        client = live_client(venue)

        result = self.finalize(client)

        self.assertTrue(result["sl_placed"])
        self.assertEqual(result["sl_order_id"], "fallback-stop")
        self.assertEqual(len(venue.created), 1)
        args, kwargs = venue.created[0]
        self.assertEqual(args[:4], (self.symbol, "market", "sell", 0.25))
        self.assertEqual(kwargs["params"]["stopLossPrice"], 96.0)
        self.assertTrue(kwargs["params"]["reduceOnly"])

    def test_unverified_attachment_and_failed_fallback_report_unprotected(self):
        venue = Venue([], create_error="stop rejected")
        client = live_client(venue)

        result = self.finalize(client)

        self.assertFalse(result["sl_placed"])
        self.assertIsNone(result["sl_order_id"])
        self.assertIn("no matching open stop", result["sl_error"])
        self.assertIn("stop rejected", result["sl_error"])
        self.assertFalse(client._local_brackets[self.symbol]["exchange_sl_set"])

    def test_verification_error_still_attempts_reduce_only_fallback(self):
        venue = Venue(fetch_error="trigger endpoint unavailable")
        client = live_client(venue)

        result = self.finalize(client, stop_attached=False)

        self.assertTrue(result["sl_placed"])
        self.assertEqual(result["sl_order_id"], "fallback-stop")
        self.assertEqual(len(venue.created), 1)

    def test_paper_path_is_unchanged_and_never_queries_venue(self):
        venue = Venue(fetch_error="must not query paper")
        client = live_client(venue)
        client.mode = "paper"

        result = self.finalize(client)

        self.assertEqual(result, {"sl_placed": True, "tp_placed": True})
        self.assertEqual(venue.fetches, [])
        self.assertEqual(venue.created, [])

    def test_direct_market_entry_reuses_proven_attached_stop(self):
        venue = DirectEntryVenue([{
            "id": "direct-attached-stop",
            "symbol": self.symbol,
            "side": "sell",
            "triggerPrice": 96.0,
        }])
        client = live_client(venue)

        result = client.place_order(
            self.symbol, Side.LONG, amount=0.25, stop_loss=96.0
        )

        self.assertEqual(len(venue.created), 1)
        self.assertEqual(
            venue.created[0][1]["params"]["stopLoss"],
            {"triggerPrice": 96.0, "triggerPriceType": "mark"},
        )
        self.assertTrue(result["sl_placed"])
        self.assertEqual(result["sl_order_id"], "direct-attached-stop")
        bracket = client._local_brackets[self.symbol]
        self.assertTrue(bracket["sl_attached"])
        self.assertEqual(bracket["sl_order_id"], "direct-attached-stop")

    def test_direct_market_entry_places_fallback_when_attachment_unproven(self):
        venue = DirectEntryVenue([])
        client = live_client(venue)

        result = client.place_order(
            self.symbol, Side.LONG, amount=0.25, stop_loss=96.0
        )

        self.assertEqual(len(venue.created), 2)
        fallback_args, fallback_kwargs = venue.created[1]
        self.assertEqual(
            fallback_args[:4], (self.symbol, "market", "sell", 0.25)
        )
        self.assertTrue(fallback_kwargs["params"]["reduceOnly"])
        self.assertEqual(fallback_kwargs["params"]["stopLossPrice"], 96.0)
        self.assertTrue(result["sl_placed"])
        self.assertEqual(result["sl_order_id"], "fallback-stop")
        self.assertFalse(client._local_brackets[self.symbol]["sl_attached"])

    def test_direct_market_entry_reports_unprotected_if_fallback_fails(self):
        venue = DirectEntryVenue([], fallback_error="fallback rejected")
        client = live_client(venue)

        result = client.place_order(
            self.symbol, Side.LONG, amount=0.25, stop_loss=96.0
        )

        self.assertFalse(result["sl_placed"])
        self.assertFalse(result["sl_trigger"])
        self.assertIsNone(result["sl_order_id"])
        self.assertIn("no matching open stop", result["sl_error"])
        self.assertIn("fallback rejected", result["sl_error"])
        self.assertFalse(client._local_brackets[self.symbol]["exchange_sl_set"])


if __name__ == "__main__":
    unittest.main()
