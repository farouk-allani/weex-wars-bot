"""WEEX AI Wars II — Exchange Client (ccxt-based) v8

Fixes:
- SL/TP stored and applied in paper mode
- Live SL/TP cached locally when exchange returns none
- Env passphrase fallback (WEEX_API_PASSPHRASE | WEEX_PASSPHRASE)
- Exchange stop/TP placement errors surfaced (not silently dropped)
"""

import ccxt
import os
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from .models import Candle, Side, OrderType, AccountState, Position

load_dotenv()


class ExchangeClient:
    """WEEX futures exchange client using ccxt."""

    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("trading", {}).get("mode", "paper")

        api_key = os.getenv("WEEX_API_KEY", "")
        api_secret = os.getenv("WEEX_API_SECRET", "")
        # Support both env names (common mismatch)
        api_passphrase = (
            os.getenv("WEEX_API_PASSPHRASE")
            or os.getenv("WEEX_PASSPHRASE")
            or ""
        )

        exchange_config = {
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
            },
        }

        self.exchange = ccxt.weex(exchange_config)
        if self.mode != "paper":
            self.exchange.set_sandbox_mode(False)

        # Paper state
        self.balance = float(config.get("backtest", {}).get("initial_capital", 10000))
        self.paper_positions: dict[str, Position] = {}
        self.paper_trades: list = []

        # Live: remember SL/TP we set (exchange position fetch often omits them)
        self._local_brackets: dict[str, dict] = {}

        # Cache
        self._candle_cache: dict[str, list[Candle]] = {}
        self._last_fetch: dict[str, float] = {}

    # ---- Market Data ----

    def fetch_candles(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> list[Candle]:
        """Fetch OHLCV candles from exchange."""
        cache_key = f"{symbol}_{timeframe}"
        ttl = 30 if timeframe in ("15m", "1h") else 60

        if cache_key in self._last_fetch:
            if time.time() - self._last_fetch[cache_key] < ttl:
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
            print(f"[Exchange] Error fetching candles for {symbol} {timeframe}: {e}")
            return self._candle_cache.get(cache_key, [])

    def fetch_ticker(self, symbol: str) -> dict:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"[Exchange] Error fetching ticker for {symbol}: {e}")
            return {}

    def fetch_funding_rate(self, symbol: str) -> float:
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return float(funding.get("fundingRate") or 0.0)
        except Exception:
            return 0.0

    # ---- Account ----

    def get_account_state(self) -> AccountState:
        if self.mode == "paper":
            self.update_paper_positions()
            unrealized = sum(p.unrealized_pnl for p in self.paper_positions.values())
            margin_used = sum(
                p.size * p.entry_price / max(p.leverage, 1)
                for p in self.paper_positions.values()
            )
            return AccountState(
                balance=self.balance,
                equity=self.balance + unrealized,
                unrealized_pnl=unrealized,
                margin_used=margin_used,
                available_margin=max(0.0, self.balance - margin_used),
                positions=list(self.paper_positions.values()),
            )

        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            positions = self.exchange.fetch_positions()

            pos_list = []
            for p in positions:
                contracts = abs(float(p.get("contracts") or 0))
                if contracts <= 0:
                    continue
                symbol = p["symbol"]
                bracket = self._local_brackets.get(symbol, {})
                side = Side.LONG if p.get("side") == "long" else Side.SHORT
                entry = float(p.get("entryPrice") or 0)
                pos_list.append(
                    Position(
                        symbol=symbol,
                        side=side,
                        entry_price=entry,
                        size=contracts,
                        leverage=int(p.get("leverage") or 1),
                        stop_loss=float(bracket.get("stop_loss") or 0),
                        take_profit=float(bracket.get("take_profit") or 0),
                        trailing_stop=bracket.get("trailing_stop"),
                        unrealized_pnl=float(p.get("unrealizedPnl") or 0),
                        highest_price=float(bracket.get("highest_price") or entry),
                        lowest_price=float(bracket.get("lowest_price") or entry),
                        strategy=str(bracket.get("strategy") or ""),
                        exchange_sl_set=bool(bracket.get("exchange_sl_set")),
                        exchange_tp_set=bool(bracket.get("exchange_tp_set")),
                    )
                )

            free = float(usdt.get("free") or 0)
            total = float(usdt.get("total") or free)
            used = float(usdt.get("used") or 0)
            return AccountState(
                balance=free,
                equity=total,
                unrealized_pnl=sum(p.unrealized_pnl for p in pos_list),
                margin_used=used,
                available_margin=free,
                positions=pos_list,
            )
        except Exception as e:
            print(f"[Exchange] Error fetching account: {e}")
            return AccountState(0, 0, 0, 0, 0)

    # ---- Trading ----

    def set_leverage(self, symbol: str, leverage: int) -> bool:
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
        strategy: str = "",
        leverage: Optional[int] = None,
    ) -> dict:
        """Place entry + optional SL/TP brackets."""
        if self.mode == "paper":
            return self._paper_order(
                symbol, side, amount, price, order_type,
                stop_loss=stop_loss, take_profit=take_profit,
                strategy=strategy, leverage=leverage,
            )

        try:
            ccxt_side = "buy" if side == Side.LONG else "sell"
            if order_type == OrderType.LIMIT and price:
                order = self.exchange.create_order(
                    symbol, "limit", ccxt_side, amount, price
                )
            else:
                order = self.exchange.create_order(
                    symbol, "market", ccxt_side, amount
                )

            sl_ok, tp_ok = False, False
            sl_err, tp_err = None, None

            if stop_loss and stop_loss > 0:
                sl_side = "sell" if side == Side.LONG else "buy"
                try:
                    self.exchange.create_order(
                        symbol, "stop_market", sl_side, amount,
                        params={"stopPrice": stop_loss, "reduceOnly": True},
                    )
                    sl_ok = True
                except Exception as e:
                    sl_err = str(e)
                    print(f"[Exchange] WARNING: SL order failed for {symbol}: {e}")

            if take_profit and take_profit > 0:
                tp_side = "sell" if side == Side.LONG else "buy"
                try:
                    self.exchange.create_order(
                        symbol, "take_profit_market", tp_side, amount,
                        params={"stopPrice": take_profit, "reduceOnly": True},
                    )
                    tp_ok = True
                except Exception as e:
                    tp_err = str(e)
                    print(f"[Exchange] WARNING: TP order failed for {symbol}: {e}")

            # Cache brackets so software management still works
            fill_price = float(
                order.get("average")
                or order.get("price")
                or price
                or 0
            )
            self._local_brackets[symbol] = {
                "stop_loss": stop_loss or 0,
                "take_profit": take_profit or 0,
                "trailing_stop": None,
                "highest_price": fill_price,
                "lowest_price": fill_price,
                "strategy": strategy,
                "exchange_sl_set": sl_ok,
                "exchange_tp_set": tp_ok,
                "side": side.value,
            }

            order = dict(order)
            order["sl_placed"] = sl_ok
            order["tp_placed"] = tp_ok
            if sl_err:
                order["sl_error"] = sl_err
            if tp_err:
                order["tp_error"] = tp_err
            return order

        except Exception as e:
            print(f"[Exchange] Error placing order: {e}")
            return {"error": str(e)}

    def update_local_brackets(self, position: Position):
        """Sync software-managed stops back into local cache (live)."""
        if self.mode == "paper":
            return
        self._local_brackets[position.symbol] = {
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "trailing_stop": position.trailing_stop,
            "highest_price": position.highest_price,
            "lowest_price": position.lowest_price,
            "strategy": position.strategy,
            "exchange_sl_set": position.exchange_sl_set,
            "exchange_tp_set": position.exchange_tp_set,
            "side": position.side.value,
        }

    def close_position(self, symbol: str) -> dict:
        if self.mode == "paper":
            if symbol not in self.paper_positions:
                return {"closed": True, "reason": "no_position"}
            pos = self.paper_positions.pop(symbol)
            ticker = self.fetch_ticker(symbol)
            exit_price = float(ticker.get("last") or pos.entry_price)
            pnl = pos.calculate_pnl(exit_price)
            self.balance += pnl
            self.paper_trades.append({"symbol": symbol, "pnl": pnl})
            return {"closed": True, "pnl": pnl, "exit_price": exit_price}

        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = abs(float(pos.get("contracts") or 0))
                if contracts > 0:
                    side = "sell" if pos.get("side") == "long" else "buy"
                    order = self.exchange.create_order(
                        symbol, "market", side, contracts,
                        params={"reduceOnly": True},
                    )
                    self._local_brackets.pop(symbol, None)
                    return order
            self._local_brackets.pop(symbol, None)
            return {"closed": True, "reason": "no_position"}
        except Exception as e:
            return {"error": str(e)}

    # ---- Paper Trading ----

    def _paper_order(
        self,
        symbol: str,
        side: Side,
        amount: float,
        price: Optional[float],
        order_type: OrderType,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = "",
        leverage: Optional[int] = None,
    ) -> dict:
        ticker = self.fetch_ticker(symbol)
        if not ticker:
            return {"error": "No ticker data"}

        fill_price = float(ticker.get("last") or price or 0)
        if fill_price <= 0:
            return {"error": "No price available"}

        if symbol in self.paper_positions:
            return {"error": "Position already exists"}

        lev = leverage or self.config.get("trading", {}).get("default_leverage", 5)
        sl = float(stop_loss or 0)
        tp = float(take_profit or 0)

        # Validate SL/TP sides
        if sl > 0:
            if side == Side.LONG and sl >= fill_price:
                sl = fill_price * 0.98
            if side == Side.SHORT and sl <= fill_price:
                sl = fill_price * 1.02
        if tp > 0:
            if side == Side.LONG and tp <= fill_price:
                tp = fill_price * 1.02
            if side == Side.SHORT and tp >= fill_price:
                tp = fill_price * 0.98

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            size=amount,
            leverage=int(lev),
            stop_loss=sl,
            take_profit=tp,
            highest_price=fill_price,
            lowest_price=fill_price,
            strategy=strategy,
            exchange_sl_set=sl > 0,
            exchange_tp_set=tp > 0,
        )
        self.paper_positions[symbol] = position

        order_id = f"paper_{int(time.time() * 1000)}"
        return {
            "id": order_id,
            "symbol": symbol,
            "side": side.value,
            "amount": amount,
            "price": fill_price,
            "status": "filled",
            "stop_loss": sl,
            "take_profit": tp,
            "sl_placed": sl > 0,
            "tp_placed": tp > 0,
        }

    def update_paper_positions(self):
        for symbol, pos in list(self.paper_positions.items()):
            ticker = self.fetch_ticker(symbol)
            if not ticker:
                continue
            current_price = float(ticker.get("last") or pos.entry_price)
            pos.unrealized_pnl = pos.calculate_pnl(current_price)
            pos.update_extremes(current_price)
