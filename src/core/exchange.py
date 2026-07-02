"""WEEX AI Wars II — Exchange Client (ccxt-based)"""

import ccxt
import os
import time
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

from .models import Candle, Side, OrderType, OrderStatus, AccountState, Position

load_dotenv()


class ExchangeClient:
    """WEEX futures exchange client using ccxt."""

    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("trading", {}).get("mode", "paper")

        # Initialize ccxt
        exchange_config = {
            "apiKey": os.getenv("WEEX_API_KEY"),
            "secret": os.getenv("WEEX_API_SECRET"),
            "password": os.getenv("WEEX_API_PASSPHRASE"),
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",  # futures
            },
        }

        if self.mode == "paper":
            self.exchange = ccxt.weex(exchange_config)
            self.balance = 10000.0  # Starting paper balance
            self.paper_positions: dict[str, Position] = {}
            self.paper_trades: list = []
        else:
            self.exchange = ccxt.weex(exchange_config)
            self.exchange.set_sandbox_mode(False)
            self.balance = 0.0

        # Cache
        self._candle_cache: dict[str, list[Candle]] = {}
        self._last_fetch: dict[str, float] = {}

    # ---- Market Data ----

    def fetch_candles(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> list[Candle]:
        """Fetch OHLCV candles from exchange."""
        cache_key = f"{symbol}_{timeframe}"

        # Cache for 30 seconds to avoid rate limits
        if cache_key in self._last_fetch:
            if time.time() - self._last_fetch[cache_key] < 30:
                return self._candle_cache.get(cache_key, [])

        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            candles = [
                Candle(
                    timestamp=datetime.utcfromtimestamp(row[0] / 1000),
                    open=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    volume=row[5],
                )
                for row in ohlcv
            ]

            self._candle_cache[cache_key] = candles
            self._last_fetch[cache_key] = time.time()
            return candles

        except Exception as e:
            print(f"[Exchange] Error fetching candles for {symbol}: {e}")
            return self._candle_cache.get(cache_key, [])

    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"[Exchange] Error fetching ticker for {symbol}: {e}")
            return {}

    def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate."""
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return funding.get("fundingRate", 0.0)
        except Exception:
            return 0.0

    # ---- Account ----

    def get_account_state(self) -> AccountState:
        """Get current account state."""
        if self.mode == "paper":
            unrealized = sum(
                p.unrealized_pnl for p in self.paper_positions.values()
            )
            return AccountState(
                balance=self.balance,
                equity=self.balance + unrealized,
                unrealized_pnl=unrealized,
                margin_used=sum(
                    p.size * p.entry_price / p.leverage
                    for p in self.paper_positions.values()
                ),
                available_margin=self.balance - sum(
                    p.size * p.entry_price / p.leverage
                    for p in self.paper_positions.values()
                ),
                positions=list(self.paper_positions.values()),
            )

        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            positions = self.exchange.fetch_positions()

            pos_list = []
            for p in positions:
                if abs(p.get("contracts", 0)) > 0:
                    pos_list.append(
                        Position(
                            symbol=p["symbol"],
                            side=Side.LONG if p["side"] == "long" else Side.SHORT,
                            entry_price=p["entryPrice"],
                            size=abs(p["contracts"]),
                            leverage=int(p.get("leverage", 1)),
                            stop_loss=0,
                            take_profit=0,
                            unrealized_pnl=p.get("unrealizedPnl", 0),
                        )
                    )

            return AccountState(
                balance=usdt.get("free", 0),
                equity=usdt.get("total", 0),
                unrealized_pnl=sum(p.unrealized_pnl for p in pos_list),
                margin_used=usdt.get("used", 0),
                available_margin=usdt.get("free", 0),
                positions=pos_list,
            )

        except Exception as e:
            print(f"[Exchange] Error fetching account: {e}")
            return AccountState(0, 0, 0, 0, 0)

    # ---- Trading ----

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            print(f"[Exchange] Error setting leverage: {e}")
            return False

    def place_order(
        self,
        symbol: str,
        side: Side,
        amount: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """Place an order."""
        if self.mode == "paper":
            return self._paper_order(symbol, side, amount, price, order_type)

        try:
            ccxt_side = "buy" if side == Side.LONG else "sell"

            if order_type == OrderType.MARKET:
                order = self.exchange.create_order(
                    symbol, "market", ccxt_side, amount
                )
            elif order_type == OrderType.LIMIT:
                order = self.exchange.create_order(
                    symbol, "limit", ccxt_side, amount, price
                )
            else:
                order = self.exchange.create_order(
                    symbol, "market", ccxt_side, amount
                )

            # Set stop-loss and take-profit if provided
            if stop_loss:
                sl_side = "sell" if side == Side.LONG else "buy"
                try:
                    self.exchange.create_order(
                        symbol, "stop_market", sl_side, amount,
                        params={"stopPrice": stop_loss, "reduceOnly": True}
                    )
                except Exception:
                    pass

            if take_profit:
                tp_side = "sell" if side == Side.LONG else "buy"
                try:
                    self.exchange.create_order(
                        symbol, "take_profit_market", tp_side, amount,
                        params={"stopPrice": take_profit, "reduceOnly": True}
                    )
                except Exception:
                    pass

            return order

        except Exception as e:
            print(f"[Exchange] Error placing order: {e}")
            return {"error": str(e)}

    def close_position(self, symbol: str) -> dict:
        """Close an open position."""
        if self.mode == "paper":
            if symbol in self.paper_positions:
                pos = self.paper_positions.pop(symbol)
                ticker = self.fetch_ticker(symbol)
                exit_price = ticker.get("last", pos.entry_price)
                pnl = pos.calculate_pnl(exit_price)
                self.balance += pnl
                return {"closed": True, "pnl": pnl}

        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if abs(pos.get("contracts", 0)) > 0:
                    side = "sell" if pos["side"] == "long" else "buy"
                    order = self.exchange.create_order(
                        symbol, "market", side, abs(pos["contracts"]),
                        params={"reduceOnly": True}
                    )
                    return order
            return {"closed": True, "reason": "no_position"}
        except Exception as e:
            return {"error": str(e)}

    # ---- Paper Trading ----

    def _paper_order(
        self, symbol: str, side: Side, amount: float,
        price: Optional[float], order_type: OrderType
    ) -> dict:
        """Simulate order execution for paper trading."""
        ticker = self.fetch_ticker(symbol)
        if not ticker:
            return {"error": "No ticker data"}

        fill_price = ticker.get("last", price or 0)
        if fill_price == 0:
            return {"error": "No price available"}

        order_id = f"paper_{int(time.time() * 1000)}"

        # Check if we already have a position
        if symbol in self.paper_positions:
            return {"error": "Position already exists", "order_id": order_id}

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            size=amount,
            leverage=self.config.get("trading", {}).get("default_leverage", 5),
            stop_loss=0,
            take_profit=0,
            highest_price=fill_price,
            lowest_price=fill_price,
        )

        self.paper_positions[symbol] = position

        return {
            "id": order_id,
            "symbol": symbol,
            "side": side.value,
            "amount": amount,
            "price": fill_price,
            "status": "filled",
        }

    def update_paper_positions(self):
        """Update unrealized PnL for paper positions."""
        for symbol, pos in self.paper_positions.items():
            ticker = self.fetch_ticker(symbol)
            if ticker:
                current_price = ticker.get("last", pos.entry_price)
                pos.unrealized_pnl = pos.calculate_pnl(current_price)
                pos.update_extremes(current_price)
