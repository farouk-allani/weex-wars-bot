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
from datetime import datetime, timedelta
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

        # Paper state — same cost model as the backtest, otherwise paper results
        # are optimistic and can't be compared against the WFO that tuned them.
        bt = config.get("backtest", {})
        self.commission_rate = float(bt.get("commission_rate", 0.0006))
        self.slippage_pct = float(bt.get("slippage_pct", 0.0005))
        # Maker fee for resting limit fills. A maker fill pays no slippage either —
        # the price is ours by construction; what we risk instead is not filling.
        exec_cfg = config.get("execution", {}) or {}
        self.maker_fee_rate = float(exec_cfg.get("maker_fee_rate", 0.0002))
        # Defaults ON: the honest simulation is the one that charges what live
        # charges. The flag exists to reproduce old runs, not to flatter new ones.
        self.paper_funding_enabled = bool(exec_cfg.get("paper_funding", True))
        # A resting limit only fills when the market trades THROUGH it — a touch
        # leaves us at the back of the queue with nothing done. Measured in ticks.
        self.fill_through_ticks = float(exec_cfg.get("paper_fill_through_ticks", 1))
        self.balance = float(bt.get("initial_capital", 10000))
        self.paper_positions: dict[str, Position] = {}
        self.paper_trades: list = []
        # Paper simulation of resting entry orders, keyed by order id. Live resting
        # orders live on the venue; this ledger only backs paper mode.
        self.paper_pending: dict[str, dict] = {}

        # Live: remember SL/TP we set (exchange position fetch often omits them)
        self._local_brackets: dict[str, dict] = {}

        # Cache
        self._candle_cache: dict[str, list[Candle]] = {}
        self._last_fetch: dict[str, float] = {}

        # Venue precision, loaded lazily on first use.
        self._markets_loaded = False

    # ---- Venue precision ----
    #
    # Sizes and prices are rounded to the venue's tradable increments *before* an
    # order is built, not left for the venue to adjust afterwards. Two reasons, and
    # the first is a hard competition requirement: the ai-log we submit must carry
    # the parameters that match the final trade request, so the number we log has to
    # be the number we sent. Second, paper rounds identically to live, so a paper
    # fill is always a size live would actually accept.

    def _ensure_markets(self) -> bool:
        if self._markets_loaded:
            return True
        try:
            self.exchange.load_markets()
            self._markets_loaded = True
        except Exception:
            self._markets_loaded = False
        return self._markets_loaded

    def normalize_amount(self, symbol: str, amount: float) -> float:
        """Round a size to the venue's amount precision. Falls back to the raw
        value if markets are unreachable — never blocks a trade."""
        if self._ensure_markets():
            try:
                return float(self.exchange.amount_to_precision(symbol, amount))
            except Exception:
                pass
        return float(amount)

    def normalize_price(self, symbol: str, price: float) -> float:
        """Round a price to the venue's tick size, same contract as above."""
        if self._ensure_markets():
            try:
                return float(self.exchange.price_to_precision(symbol, price))
            except Exception:
                pass
        return float(price)

    def _tick_size(self, symbol: str, ref_price: float) -> float:
        """One price increment for `symbol`, in quote units.

        ccxt reports price precision as either a tick (0.01) or a count of decimal
        places (2), decided by the exchange's `precisionMode` — and the two forms
        collide at the value 1, which is a $1 tick under one reading and $0.1 under
        the other. Read the mode rather than guess from the magnitude.

        Falls back to one basis point of the reference price when markets are
        unreachable. The fallback must never be 0, or the trade-through test
        silently degrades back into the touch test it replaced.
        """
        if self._ensure_markets():
            try:
                prec = self.exchange.market(symbol)["precision"]["price"]
                if prec is not None:
                    if self.exchange.precisionMode == ccxt.DECIMAL_PLACES:
                        return 10.0 ** -int(prec)
                    tick = float(prec)
                    if tick > 0:
                        return tick
            except Exception:
                pass
        return abs(ref_price) * 1e-4

    # ---- Market Data ----

    def fetch_candles(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> list[Candle]:
        """Fetch OHLCV candles from exchange.

        The cache is keyed on symbol+timeframe but NOT on `limit`, so it must only
        serve a request it can actually satisfy. It previously ignored `limit`
        entirely, which had a silent and expensive consequence: `_manage_positions`
        runs first each cycle and asks for 30 1h candles per open position, so the
        AI context loop's later request for 200 got the cached 30, failed its
        `len(candles) < 100` guard, and every held symbol was dropped from the
        model's market data. Measured 2026-07-27: 42 of 42 open positions had no
        market data in the same context — the model could see its entry and stop
        but not the tape it was deciding against. Serving the tail of a longer
        cached series is fine; serving a shorter one is not.
        """
        cache_key = f"{symbol}_{timeframe}"
        ttl = 30 if timeframe in ("15m", "1h") else 60

        cached = self._candle_cache.get(cache_key, [])
        if (
            cache_key in self._last_fetch
            and time.time() - self._last_fetch[cache_key] < ttl
            and len(cached) >= limit
        ):
            return cached[-limit:]

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
                        # Live positions are rebuilt from the API every cycle, so
                        # anything the exchange doesn't know has to survive here —
                        # otherwise a banked partial is forgotten before the close.
                        partial_take_profit=bracket.get("partial_take_profit"),
                        partial_fraction=float(bracket.get("partial_fraction") or 0.5),
                        partial_taken=bool(bracket.get("partial_taken")),
                        initial_size=float(bracket.get("initial_size") or contracts),
                        realized_pnl=float(bracket.get("realized_pnl") or 0),
                        entry_fee=float(bracket.get("entry_fee") or 0),
                        fees_paid=float(bracket.get("fees_paid") or 0),
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
        # Paper mode is local sim — no API credentials needed
        if self.mode == "paper":
            return True
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
        # Same contract as place_entry_limit: round before recording, so the
        # ai-log's quantity/price match the submitted request.
        amount = self.normalize_amount(symbol, amount)
        if amount <= 0:
            return {"error": "amount rounds to zero at venue precision"}
        if price:
            price = self.normalize_price(symbol, price)

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

            sl_ok, tp_ok, sl_err, tp_err = self._create_live_brackets(
                symbol, side, amount, stop_loss, take_profit
            )

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
                "initial_size": amount,
                "entry_fee": amount * fill_price * self.commission_rate,
                "fees_paid": amount * fill_price * self.commission_rate,
                "realized_pnl": 0.0,
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

    def _create_live_brackets(
        self,
        symbol: str,
        side: Side,
        amount: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> tuple[bool, bool, Optional[str], Optional[str]]:
        """Place venue-side SL/TP reduce-only orders. Failures are surfaced, not
        raised — the caller keeps the position and falls back to software stops."""
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

        return sl_ok, tp_ok, sl_err, tp_err

    # ---- Maker (post-only) entries ----
    #
    # Why this path exists: a market entry pays taker fee + slippage (~0.11% of
    # notional per side); a resting post-only limit pays the maker fee and no
    # spread. Measured round-trip cost at market was 0.22% against a best measured
    # edge of ~0.13%/trade — execution is the difference between negative and
    # roughly breakeven. So entries rest at the touch and the engine reprices or
    # abandons; it never crosses the spread to chase a trade.

    def touch_price(self, symbol: str, side: Side) -> float:
        """Best passive price: the bid for a buy, the ask for a sell.

        Falls back to last when the venue omits book data. Never returns a price
        that crosses the spread."""
        t = self.fetch_ticker(symbol)
        if not t:
            return 0.0
        if side == Side.LONG:
            return float(t.get("bid") or t.get("last") or 0)
        return float(t.get("ask") or t.get("last") or 0)

    def place_entry_limit(
        self,
        symbol: str,
        side: Side,
        amount: float,
        limit_price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        strategy: str = "",
        leverage: Optional[int] = None,
        partial_take_profit: Optional[float] = None,
        partial_fraction: float = 0.5,
    ) -> dict:
        """Rest a post-only entry at limit_price. Returns {id, status} or {error}.

        The bracket levels travel WITH the pending order so a fill can never
        produce a position without a stop, even across a restart."""
        if limit_price <= 0 or amount <= 0:
            return {"error": "invalid limit price or amount"}

        # Round to venue increments before anything records these numbers: the
        # pending order, the resulting position, and the audited ai-log must all
        # agree with what was actually submitted.
        amount = self.normalize_amount(symbol, amount)
        limit_price = self.normalize_price(symbol, limit_price)
        if amount <= 0:
            return {"error": "amount rounds to zero at venue precision"}

        if self.mode == "paper":
            if symbol in self.paper_positions:
                return {"error": "Position already exists"}
            order_id = f"pending_{symbol.split('/')[0]}_{int(time.time() * 1000)}"
            self.paper_pending[order_id] = {
                "id": order_id,
                "symbol": symbol,
                "side": side.value,
                "amount": amount,
                "limit_price": limit_price,
                "stop_loss": float(stop_loss or 0),
                "take_profit": float(take_profit or 0),
                "strategy": strategy,
                "leverage": int(leverage or self.config.get("trading", {}).get("default_leverage", 5)),
                "partial_take_profit": partial_take_profit,
                "partial_fraction": partial_fraction,
                "created": time.time(),
            }
            return {"id": order_id, "status": "open", "limit_price": limit_price}

        try:
            ccxt_side = "buy" if side == Side.LONG else "sell"
            order = self.exchange.create_order(
                symbol, "limit", ccxt_side, amount, limit_price,
                params={"postOnly": True},
            )
            return {
                "id": str(order.get("id")),
                "status": order.get("status") or "open",
                "limit_price": limit_price,
            }
        except Exception as e:
            # A post-only order that would cross is rejected by the venue — that is
            # the mechanism working, not a fault. The engine simply retries at the
            # new touch on its next pass.
            return {"error": str(e)}

    def check_entry_fill(self, order_id: str, symbol: str) -> dict:
        """Poll one resting entry. Returns {status: open|filled|gone, fill_price,
        filled_amount}.

        Paper fill rule: the market must trade THROUGH our limit by
        `execution.paper_fill_through_ticks` before we count a fill — a buy needs
        last <= limit - ticks, a sell last >= limit + ticks.

        Until 2026-08-03 a mere touch filled us, in full. That is the optimistic end
        of the queue: at the touch we are behind everyone already resting at that
        price, and the overwhelming majority of touches reverse without clearing the
        book down to us. It inflated the fill rate (the Aug 1 audit's 12/14 "rested"
        is an artifact of it) and, worse, it granted free entries precisely at local
        reversals — the touches that immediately turn are the ones a real queue does
        NOT give you, and they are the best-looking entries in the sample.

        Trade-through is still not a queue model; it is the pessimistic bracket. The
        truth sits between the two, and the honest thing is to be measured by the
        bracket that cannot flatter us. Price stays honest either way: the fill is at
        OUR limit, never an assumed improvement. Polling is 60s, so moves that spike
        through and recover between polls are missed — conservative in the same
        direction."""
        if self.mode == "paper":
            pending = self.paper_pending.get(order_id)
            if not pending:
                return {"status": "gone", "fill_price": 0.0, "filled_amount": 0.0}
            ticker = self.fetch_ticker(symbol)
            last = float(ticker.get("last") or 0) if ticker else 0.0
            if last <= 0:
                return {"status": "open", "fill_price": 0.0, "filled_amount": 0.0}

            limit = float(pending["limit_price"])
            side = Side(pending["side"])
            through = self.fill_through_ticks * self._tick_size(symbol, limit)
            hit = (
                last <= limit - through
                if side == Side.LONG
                else last >= limit + through
            )
            if not hit:
                return {"status": "open", "fill_price": 0.0, "filled_amount": 0.0}

            if symbol in self.paper_positions:
                # Should be unreachable (engine holds one pending per symbol), but a
                # duplicate position would corrupt the ledger — drop the order.
                self.paper_pending.pop(order_id, None)
                return {"status": "gone", "fill_price": 0.0, "filled_amount": 0.0}

            self.paper_pending.pop(order_id, None)
            self._open_paper_position(
                symbol=symbol,
                side=side,
                amount=float(pending["amount"]),
                fill_price=limit,
                fee_rate=self.maker_fee_rate,
                stop_loss=float(pending.get("stop_loss") or 0),
                take_profit=float(pending.get("take_profit") or 0),
                strategy=pending.get("strategy") or "",
                leverage=int(pending.get("leverage") or 5),
                partial_take_profit=pending.get("partial_take_profit"),
                partial_fraction=float(pending.get("partial_fraction") or 0.5),
            )
            return {
                "status": "filled",
                "fill_price": limit,
                "filled_amount": float(pending["amount"]),
            }

        try:
            order = self.exchange.fetch_order(order_id, symbol)
            status = str(order.get("status") or "open").lower()
            filled = float(order.get("filled") or 0)
            fill_price = float(order.get("average") or order.get("price") or 0)
            if status == "closed":
                return {"status": "filled", "fill_price": fill_price, "filled_amount": filled}
            if status in ("canceled", "cancelled", "expired", "rejected"):
                return {"status": "gone", "fill_price": fill_price, "filled_amount": filled}
            return {"status": "open", "fill_price": fill_price, "filled_amount": filled}
        except Exception as e:
            # fetch_order unsupported or transient failure: infer from open orders,
            # then from the position book. Anything still ambiguous stays "open" —
            # the next cycle retries rather than guessing.
            try:
                open_orders = self.exchange.fetch_open_orders(symbol)
                if any(str(o.get("id")) == order_id for o in open_orders):
                    return {"status": "open", "fill_price": 0.0, "filled_amount": 0.0}
                positions = self.exchange.fetch_positions([symbol])
                for p in positions:
                    if abs(float(p.get("contracts") or 0)) > 0:
                        return {
                            "status": "filled",
                            "fill_price": float(p.get("entryPrice") or 0),
                            "filled_amount": abs(float(p.get("contracts") or 0)),
                        }
                return {"status": "gone", "fill_price": 0.0, "filled_amount": 0.0}
            except Exception:
                print(f"[Exchange] check_entry_fill failed for {symbol}: {e}")
                return {"status": "open", "fill_price": 0.0, "filled_amount": 0.0}

    def cancel_entry(self, order_id: str, symbol: str) -> dict:
        """Cancel a resting entry. Reports any amount that filled before the cancel
        landed so the caller can bracket the partial position instead of orphaning it."""
        if self.mode == "paper":
            existed = self.paper_pending.pop(order_id, None) is not None
            return {"cancelled": existed, "filled_amount": 0.0}

        filled = 0.0
        try:
            self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            print(f"[Exchange] cancel_entry {symbol} {order_id}: {e}")
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            filled = float(order.get("filled") or 0)
        except Exception:
            pass
        return {"cancelled": True, "filled_amount": filled}

    def finalize_entry_fill(
        self,
        symbol: str,
        side: Side,
        amount: float,
        fill_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "",
        partial_take_profit: Optional[float] = None,
        partial_fraction: float = 0.5,
    ) -> dict:
        """Attach brackets to a just-filled maker entry.

        Paper positions were already built with their brackets at fill time, so
        this is live-only work: venue SL/TP orders plus the local bracket cache
        that software management reads."""
        if self.mode == "paper":
            return {"sl_placed": True, "tp_placed": True}

        sl_ok, tp_ok, sl_err, tp_err = self._create_live_brackets(
            symbol, side, amount, stop_loss, take_profit
        )
        entry_fee = amount * fill_price * self.maker_fee_rate
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
            "initial_size": amount,
            "entry_fee": entry_fee,
            "fees_paid": entry_fee,
            "realized_pnl": 0.0,
            "partial_take_profit": partial_take_profit,
            "partial_fraction": partial_fraction,
        }
        out = {"sl_placed": sl_ok, "tp_placed": tp_ok}
        if sl_err:
            out["sl_error"] = sl_err
        if tp_err:
            out["tp_error"] = tp_err
        return out

    def _open_paper_position(
        self,
        symbol: str,
        side: Side,
        amount: float,
        fill_price: float,
        fee_rate: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        strategy: str = "",
        leverage: Optional[int] = None,
        partial_take_profit: Optional[float] = None,
        partial_fraction: float = 0.5,
    ) -> Position:
        """Book a paper position at an explicit fill price and fee rate — shared by
        market fills (slippage + taker) and limit fills (own price + maker)."""
        entry_fee = amount * fill_price * fee_rate
        self.balance -= entry_fee

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
            initial_size=amount,
            entry_fee=entry_fee,
            fees_paid=entry_fee,
            partial_take_profit=partial_take_profit,
            partial_fraction=partial_fraction,
            # Anchor at the settlement preceding this fill, so the first charge is
            # the next boundary crossed — never the one that already passed before
            # the position existed.
            last_funding_at=self._funding_boundary(datetime.utcnow()),
        )
        self.paper_positions[symbol] = position
        return position

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
            "partial_take_profit": position.partial_take_profit,
            "partial_fraction": position.partial_fraction,
            "partial_taken": position.partial_taken,
            "initial_size": position.initial_size or position.size,
            "realized_pnl": position.realized_pnl,
            "entry_fee": position.entry_fee,
            "fees_paid": position.fees_paid,
        }

    def close_position(self, symbol: str) -> dict:
        if self.mode == "paper":
            if symbol not in self.paper_positions:
                return {"closed": True, "reason": "no_position"}
            pos = self.paper_positions.pop(symbol)
            ticker = self.fetch_ticker(symbol)
            mark = float(ticker.get("last") or pos.entry_price)
            # Settle funding owed up to this instant before the position leaves the
            # book, or a close that lands just after a boundary escapes the charge.
            self._settle_paper_funding(pos, mark, datetime.utcnow())
            exit_price = self.apply_slippage(mark, pos.side, is_exit=True)
            fee = pos.size * exit_price * self.commission_rate
            pnl = pos.calculate_pnl(exit_price) - fee
            self.balance += pnl
            # `pnl` stays execution-only; funding already moved the balance as it
            # accrued. Reporting it separately keeps the exit research honest —
            # a hold-longer rule pays more carry without touching the fee line.
            self.paper_trades.append(
                {"symbol": symbol, "pnl": pnl, "funding": pos.funding_paid}
            )
            return {
                "closed": True,
                "pnl": pnl,
                "exit_price": exit_price,
                "fee": fee,
                "funding": pos.funding_paid,
            }

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

    def apply_slippage(self, price: float, side: Side, is_exit: bool = False) -> float:
        """Fill against us: buys fill higher, sells fill lower."""
        buying = (side == Side.LONG) != is_exit
        return price * (1 + self.slippage_pct) if buying else price * (1 - self.slippage_pct)

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

        mark = float(ticker.get("last") or price or 0)
        if mark <= 0:
            return {"error": "No price available"}

        if symbol in self.paper_positions:
            return {"error": "Position already exists"}

        fill_price = self.apply_slippage(mark, side)
        position = self._open_paper_position(
            symbol=symbol,
            side=side,
            amount=amount,
            fill_price=fill_price,
            fee_rate=self.commission_rate,
            stop_loss=float(stop_loss or 0),
            take_profit=float(take_profit or 0),
            strategy=strategy,
            leverage=leverage,
        )

        order_id = f"paper_{int(time.time() * 1000)}"
        return {
            "id": order_id,
            "symbol": symbol,
            "side": side.value,
            "amount": amount,
            "price": fill_price,
            "status": "filled",
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "sl_placed": position.stop_loss > 0,
            "tp_placed": position.take_profit > 0,
            "entry_fee": position.entry_fee,
        }

    # ---- Funding ----
    #
    # Perpetuals settle funding at 00:00/08:00/16:00 UTC. Until 2026-08-03 the paper
    # ledger fetched the rate, showed it to the model as context, and never charged
    # it — so a paper position was free to hold and a live one was not. That biased
    # every result in one direction (longs in a positive-funding regime looked better
    # than they were), and it invalidated the one surviving research candidate
    # outright: a funding-carry book measured in a simulator that does not charge
    # funding is measuring nothing.

    FUNDING_HOURS = (0, 8, 16)

    @classmethod
    def _funding_boundary(cls, when: datetime) -> datetime:
        """Most recent settlement instant at or before `when`.

        Hour 0 is a boundary, so some element always qualifies — no wrap-to-yesterday
        case exists.
        """
        hour = max(h for h in cls.FUNDING_HOURS if h <= when.hour)
        return when.replace(hour=hour, minute=0, second=0, microsecond=0)

    @classmethod
    def _boundaries_between(cls, after: datetime, until: datetime) -> list[datetime]:
        """Settlements strictly after `after` and at or before `until`."""
        out: list[datetime] = []
        b = cls._funding_boundary(until)
        while b > after:
            out.append(b)
            b = cls._funding_boundary(b - timedelta(seconds=1))
        return sorted(out)

    def _settle_paper_funding(self, pos: Position, mark: float, now: datetime) -> float:
        """Charge every funding settlement this position has slept through.

        Longs pay a positive rate, shorts receive it. Settled straight to balance,
        the way the venue does it — not folded into unrealised PnL, because it is
        cash that has already moved and does not come back if price reverses.
        """
        if not self.paper_funding_enabled or mark <= 0:
            return 0.0
        anchor = pos.last_funding_at or self._funding_boundary(pos.opened_at)
        due = self._boundaries_between(anchor, now)
        if not due:
            return 0.0

        # One rate lookup covers every boundary owed. Rates are only observable as
        # the venue's current estimate, so a bot that was down for two settlements
        # prices both at today's rate — approximate, and far closer than zero.
        rate = self.fetch_funding_rate(pos.symbol)
        direction = 1.0 if pos.side == Side.LONG else -1.0
        flow = rate * abs(mark * pos.size) * direction * len(due)

        self.balance -= flow
        pos.funding_paid += flow
        pos.last_funding_at = due[-1]
        if abs(flow) > 0:
            print(
                f"[Exchange] funding {pos.symbol} {pos.side.value}: "
                f"{'paid' if flow > 0 else 'received'} {abs(flow):.4f} "
                f"({len(due)} settlement(s) @ {rate:+.6f})"
            )
        return flow

    def update_paper_positions(self):
        now = datetime.utcnow()
        for symbol, pos in list(self.paper_positions.items()):
            ticker = self.fetch_ticker(symbol)
            if not ticker:
                continue
            current_price = float(ticker.get("last") or pos.entry_price)
            self._settle_paper_funding(pos, current_price, now)
            pos.unrealized_pnl = pos.calculate_pnl(current_price)
            pos.update_extremes(current_price)

    def to_state(self) -> dict:
        """Paper ledger for restart recovery (live mode reads truth from the API)."""
        if self.mode != "paper":
            return {}
        return {
            "balance": self.balance,
            # Resting entries survive a restart with their brackets intact — a fill
            # after recovery must still produce a stopped position.
            "pending": list(self.paper_pending.values()),
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "entry_price": p.entry_price,
                    "size": p.size,
                    "leverage": p.leverage,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "trailing_stop": p.trailing_stop,
                    "opened_at": p.opened_at.isoformat(),
                    "highest_price": p.highest_price,
                    "lowest_price": p.lowest_price,
                    "strategy": p.strategy,
                    "partial_take_profit": p.partial_take_profit,
                    "partial_fraction": p.partial_fraction,
                    "partial_taken": p.partial_taken,
                    "initial_size": p.initial_size,
                    "realized_pnl": p.realized_pnl,
                    "entry_fee": p.entry_fee,
                    "fees_paid": p.fees_paid,
                    "funding_paid": p.funding_paid,
                    # Without this the anchor resets to boot time on restart and
                    # every settlement slept through gets skipped — free carry,
                    # which is the bug this whole change exists to remove.
                    "last_funding_at": (
                        p.last_funding_at.isoformat() if p.last_funding_at else None
                    ),
                }
                for p in self.paper_positions.values()
            ],
        }

    def load_state(self, state: dict) -> None:
        if self.mode != "paper" or not state:
            return
        if state.get("balance") is not None:
            self.balance = float(state["balance"])
        for raw in state.get("pending") or []:
            try:
                self.paper_pending[str(raw["id"])] = dict(raw)
            except Exception as e:
                print(f"[Exchange] Could not restore pending order: {e}")
        for raw in state.get("positions") or []:
            try:
                self.paper_positions[raw["symbol"]] = Position(
                    symbol=raw["symbol"],
                    side=Side(raw["side"]),
                    entry_price=float(raw["entry_price"]),
                    size=float(raw["size"]),
                    leverage=int(raw["leverage"]),
                    stop_loss=float(raw.get("stop_loss") or 0),
                    take_profit=float(raw.get("take_profit") or 0),
                    trailing_stop=raw.get("trailing_stop"),
                    opened_at=datetime.fromisoformat(
                        str(raw["opened_at"]).replace("Z", "")
                    ),
                    highest_price=float(raw.get("highest_price") or 0),
                    lowest_price=float(raw.get("lowest_price") or float("inf")),
                    strategy=raw.get("strategy") or "",
                    partial_take_profit=raw.get("partial_take_profit"),
                    partial_fraction=float(raw.get("partial_fraction") or 0.5),
                    partial_taken=bool(raw.get("partial_taken")),
                    initial_size=float(raw.get("initial_size") or 0),
                    realized_pnl=float(raw.get("realized_pnl") or 0),
                    entry_fee=float(raw.get("entry_fee") or 0),
                    fees_paid=float(raw.get("fees_paid") or 0),
                    funding_paid=float(raw.get("funding_paid") or 0),
                    last_funding_at=(
                        datetime.fromisoformat(
                            str(raw["last_funding_at"]).replace("Z", "")
                        )
                        if raw.get("last_funding_at")
                        else None
                    ),
                )
            except Exception as e:
                print(f"[Exchange] Could not restore paper position: {e}")

    def snapshot_for_dashboard(self) -> dict:
        """Serializable account snapshot for the monitoring UI."""
        account = self.get_account_state()
        positions = []
        for p in account.positions:
            positions.append({
                "symbol": p.symbol,
                "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                "entry_price": p.entry_price,
                "size": p.size,
                "leverage": p.leverage,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "trailing_stop": p.trailing_stop,
                "unrealized_pnl": p.unrealized_pnl,
                "strategy": getattr(p, "strategy", "") or "",
                "partial_taken": getattr(p, "partial_taken", False),
                "partial_take_profit": getattr(p, "partial_take_profit", None),
                "opened_at": p.opened_at.isoformat() if getattr(p, "opened_at", None) else None,
            })
        return {
            "mode": self.mode,
            "balance": account.balance,
            "equity": account.equity,
            "unrealized_pnl": account.unrealized_pnl,
            "margin_used": account.margin_used,
            "available_margin": account.available_margin,
            "open_positions": len(positions),
            "positions": positions,
        }
