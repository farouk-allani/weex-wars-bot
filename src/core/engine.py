"""WEEX AI Wars II — Main Trading Engine v8.4

- HTF data, adaptive strategy scores
- State persistence across restarts
- Partial take-profit handling
- File logging
"""

import logging
import time
import yaml
import signal as sig
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .exchange import ExchangeClient
from .models import Side, Signal, Position, TradeResult
from ..strategies.composite import CompositeStrategy
from ..strategies.edges import EdgeStrategies
from ..risk.manager import RiskManager
from ..indicators.technical import calculate_atr
from ..utils.logger import setup_logger
from ..utils.state import save_state, load_state, DEFAULT_STATE_PATH
from ..ai import AITrader, DecisionLog, DeepSeekClient, build_context
from ..ai.context import symbol_snapshot
import numpy as np  # noqa: E402

console = Console()


class TradingEngine:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.exchange = ExchangeClient(self.config)
        self.strategy = CompositeStrategy(self.config)
        self.risk = RiskManager(self.config)
        self.logger = setup_logger(self.config)
        self.running = False
        self.cycle_count = 0
        self.state_path = Path(
            self.config.get("logging", {}).get("state_file", str(DEFAULT_STATE_PATH))
        )

        # Restore state
        state = load_state(self.state_path)
        if state:
            self.risk.load_state(state.get("risk") or {})
            paper_state = state.get("paper") or {}
            if not paper_state and (state.get("account") or {}).get("mode") == "paper":
                # State written before the paper ledger was persisted: the dashboard
                # snapshot is the only record of the balance, so seed from it.
                paper_state = {"balance": (state.get("account") or {}).get("balance")}
            self.exchange.load_state(paper_state)
            lt = state.get("last_trade_time") or {}
            for k, v in lt.items():
                try:
                    self.strategy.last_trade_time[k] = datetime.fromisoformat(v.replace("Z", ""))
                except Exception:
                    pass
            self.logger.info(
                "Restored bot state from %s (balance=%.2f open=%d)",
                self.state_path,
                self.exchange.balance,
                len(self.exchange.paper_positions),
            )

        self.strategy.sync_scores_from_risk(self.risk)

        # --- AI decision layer ---
        ai_cfg = self.config.get("ai", {}) or {}
        self.ai: AITrader | None = None
        self.decision_log: DecisionLog | None = None
        self.edges = EdgeStrategies(self.config)
        # Which AI decision opened which position, so outcomes link back to reasoning.
        self.position_decisions: dict[str, str] = (state or {}).get("position_decisions") or {}
        self._last_ai_call: datetime | None = None
        self.ai_interval_min = float(ai_cfg.get("decision_interval_minutes", 60))

        if ai_cfg.get("enabled", False):
            self.decision_log = DecisionLog(ai_cfg.get("log_file", "logs/ai_decisions.jsonl"))
            self.ai = AITrader(self.config, DeepSeekClient(self.config), self.decision_log)
            self.logger.info(
                "AI decision layer active: model=%s interval=%smin",
                self.ai.client.model, self.ai_interval_min,
            )

        sig.signal(sig.SIGINT, self._shutdown)
        sig.signal(sig.SIGTERM, self._shutdown)

    def run(self):
        self.running = True
        symbols = self.config.get("trading", {}).get("symbols", ["BTC/USDT:USDT"])
        timeframe = self.config.get("trading", {}).get("timeframe", "1h")
        htf = self.config.get("trading", {}).get("higher_timeframe", "4h")
        lookback = self.config.get("trading", {}).get("lookback_periods", 100)
        htf_lookback = self.config.get("trading", {}).get("htf_lookback", 80)

        pure = self.config.get("competition", {}).get("pure_edge", False)
        console.print(Panel.fit(
            "[bold green]WEEX AI Wars II — Trading Bot v8.5[/]\n"
            f"Mode: [yellow]{self.config['trading']['mode']}[/] | "
            f"Profile: [cyan]{'pure_edge' if pure else 'competition'}[/]\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Timeframes: {timeframe} + {htf}\n"
            f"Max Drawdown: {self.risk.max_drawdown:.0%}\n"
            f"Risk/Trade: {self.risk.max_risk_per_trade:.1%}\n"
            f"Features: wick-MR, partial runners, adaptive weights, state save",
            title="Bot Started",
        ))
        self.logger.info(
            "Bot start mode=%s symbols=%s",
            self.config["trading"]["mode"],
            symbols,
        )

        leverage = self.config.get("trading", {}).get("default_leverage", 5)
        for symbol in symbols:
            self.exchange.set_leverage(symbol, leverage)

        while self.running:
            try:
                self.cycle_count += 1
                self._run_cycle(symbols, timeframe, htf, lookback, htf_lookback)
                self._display_status()
                # Always snapshot for dashboard (open positions + equity)
                self._persist_state()

                sleep_time = 60 if timeframe == "1h" else 30
                for _ in range(sleep_time):
                    if not self.running:
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error in cycle: {e}[/]")
                self.logger.exception("Cycle error: %s", e)
                time.sleep(30)

        self._persist_state()
        self._cleanup()

    def _run_cycle(self, symbols, timeframe, htf, lookback, htf_lookback):
        if self.ai:
            return self._run_ai_cycle(symbols, timeframe, htf, lookback, htf_lookback)
        return self._run_rules_cycle(symbols, timeframe, htf, lookback, htf_lookback)

    def _run_rules_cycle(self, symbols, timeframe, htf, lookback, htf_lookback):
        account = self.exchange.get_account_state()
        self.strategy.sync_scores_from_risk(self.risk)

        can_trade, reason = self.risk.can_trade(account)
        if not can_trade:
            console.print(f"[yellow]Trading blocked: {reason}[/]")
            # A full book is routine, not a warning — logging it at WARNING every
            # cycle buries the blocks that actually matter (kill-switch, cooldown).
            routine = reason.startswith("Max positions")
            self.logger.log(
                logging.INFO if routine else logging.WARNING,
                "Trading blocked: %s", reason,
            )
            self._manage_positions(account)
            return

        self._manage_positions(account)
        account = self.exchange.get_account_state()

        existing = [(p.symbol, p.side.value) for p in account.positions]
        symbol_weights = {s: self.risk.get_pair_weight(s) for s in symbols}
        sorted_symbols = sorted(symbols, key=lambda s: symbol_weights[s], reverse=True)

        for symbol in sorted_symbols:
            try:
                if any(p.symbol == symbol for p in account.positions):
                    continue
                if len(account.positions) >= self.risk.max_open_positions:
                    break

                candles = self.exchange.fetch_candles(symbol, timeframe, lookback)
                if len(candles) < 100:
                    continue

                htf_candles = self.exchange.fetch_candles(symbol, htf, htf_lookback)
                funding_rate = self.exchange.fetch_funding_rate(symbol)

                signal = self.strategy.analyze(
                    symbol, candles, funding_rate, existing,
                    higher_tf_candles=htf_candles if htf_candles else None,
                )
                if signal is None:
                    continue

                can_open, why = self.risk.can_open(signal, account)
                if not can_open:
                    console.print(f"[yellow]Skipping {symbol}: {why}[/]")
                    self.logger.info("SKIP %s: %s", symbol, why)
                    continue

                pair_weight = self.risk.get_pair_weight(symbol)
                size = self.risk.calculate_position_size(signal, account, pair_weight)
                if size <= 0:
                    continue

                self._execute_trade(signal, size, pair_weight)
                account = self.exchange.get_account_state()
                existing = [(p.symbol, p.side.value) for p in account.positions]

            except Exception as e:
                console.print(f"[red]Error analyzing {symbol}: {e}[/]")
                self.logger.exception("Analyze %s: %s", symbol, e)

    def _run_ai_cycle(self, symbols, timeframe, htf, lookback, htf_lookback):
        account = self.exchange.get_account_state()

        # Stops never wait on the model. Position management runs every cycle (60s);
        # the model is consulted on its own, much slower, cadence.
        self._manage_positions(account)
        account = self.exchange.get_account_state()

        if not self._ai_due():
            return

        can_trade, reason = self.risk.can_trade(account)

        market = []
        atrs: dict[str, float] = {}
        prices: dict[str, float] = {}
        for symbol in symbols:
            try:
                candles = self.exchange.fetch_candles(symbol, timeframe, lookback)
                if len(candles) < 100:
                    continue
                htf_candles = self.exchange.fetch_candles(symbol, htf, htf_lookback)
                funding = self.exchange.fetch_funding_rate(symbol)

                highs = np.array([c.high for c in candles])
                lows = np.array([c.low for c in candles])
                closes = np.array([c.close for c in candles])
                atrs[symbol] = float(calculate_atr(highs, lows, closes)[-1])
                prices[symbol] = float(closes[-1])

                edges = self.edges.analyze_all_edges(
                    symbol, candles, funding, higher_tf_candles=htf_candles or None
                )
                market.append(
                    symbol_snapshot(symbol, candles, funding, htf_candles or None, edges)
                )
            except Exception as e:
                self.logger.exception("AI context build failed for %s: %s", symbol, e)

        if not market:
            self.logger.warning("AI cycle skipped: no market data")
            return

        context = build_context(
            symbols_data=market,
            account=account,
            risk=self.risk,
            recent_trades=self.risk.trade_history,
            competition=self._competition_context(),
        )
        # Tell the model plainly when it may not open anything, so it reasons about
        # exits and holds instead of proposing entries that will be thrown away.
        if not can_trade:
            context["hard_limits"]["entries_blocked"] = reason

        console.print(f"[cyan]Consulting {self.ai.client.model}...[/]")
        decisions, assessment, decision_id = self.ai.decide(context)
        self._last_ai_call = datetime.utcnow()

        if assessment:
            console.print(f"[dim]{assessment}[/]")
        self.logger.info("AI decision_id=%s assessment=%s", decision_id, assessment[:200])

        allowed = set(symbols)
        # Exits first: closing frees a slot that an entry below may want.
        for d in decisions:
            if str(d.get("action", "")).lower() != "close":
                continue
            sym = d.get("symbol")
            pos = next((p for p in account.positions if p.symbol == sym), None)
            if not pos:
                continue
            price = prices.get(sym) or pos.entry_price
            console.print(f"[yellow]AI closing {sym}: {d.get('rationale', '')[:100]}[/]")
            self.logger.info("AI_CLOSE %s reason=%s", sym, str(d.get("rationale"))[:200])
            self._close_position(pos, price, "ai_close")

        if not can_trade:
            console.print(f"[yellow]Entries blocked: {reason}[/]")
            self.logger.info("AI entries blocked: %s", reason)
            return

        account = self.exchange.get_account_state()
        for d in decisions:
            if str(d.get("action", "")).lower() not in ("long", "short"):
                continue
            sym = str(d.get("symbol") or "")
            if sym not in prices:
                continue
            if any(p.symbol == sym for p in account.positions):
                continue
            if len(account.positions) >= self.risk.max_open_positions:
                break

            signal, why = self.ai.to_signal(
                d, sym, prices[sym], atrs.get(sym, 0.0), allowed
            )
            if signal is None:
                # A rejected proposal is signal about the model, so keep it visible.
                console.print(f"[dim]AI {sym} rejected: {why}[/]")
                self.logger.info("AI_REJECT %s: %s", sym, why)
                continue

            ok, why = self.risk.can_open(signal, account)
            if not ok:
                console.print(f"[yellow]AI {sym} vetoed: {why}[/]")
                self.logger.info("AI_VETO %s: %s", sym, why)
                continue

            pair_weight = self.risk.get_pair_weight(sym)
            size = self.risk.calculate_position_size(signal, account, pair_weight)
            if size <= 0:
                self.logger.info("AI_VETO %s: size below minimum", sym)
                continue

            self._execute_trade(signal, size, pair_weight, decision_id=decision_id)
            account = self.exchange.get_account_state()

    def _ai_due(self) -> bool:
        if self._last_ai_call is None:
            return True
        elapsed_min = (datetime.utcnow() - self._last_ai_call).total_seconds() / 60
        return elapsed_min >= self.ai_interval_min

    def _competition_context(self) -> dict:
        """Trade count and clock — the model should know it needs 10 trades to qualify."""
        comp = self.config.get("competition", {}) or {}
        real = [t for t in self.risk.trade_history if not self.risk.is_keepalive(t.strategy)]
        ctx = {
            "ranking_metric": "cumulative PnL",
            "trades_executed": len(real),
            "minimum_trades_required": comp.get("min_trades", 10),
            "note": (
                "Fewer than the minimum trades means disqualification, but forcing "
                "low-quality trades to hit a count is a losing play. Take good ones."
            ),
        }
        ends_at = comp.get("ends_at")
        if ends_at:
            try:
                remaining = datetime.fromisoformat(str(ends_at)) - datetime.utcnow()
                ctx["hours_remaining"] = round(remaining.total_seconds() / 3600, 1)
            except Exception:
                pass
        return ctx

    def _execute_trade(
        self,
        signal: Signal,
        size: float,
        pair_weight: float = 1.0,
        decision_id: str | None = None,
    ):
        console.print(f"\n[bold cyan]Signal: {signal.side.value.upper()} {signal.symbol}[/]")
        console.print(f"   Strategy: {signal.strategy}")
        console.print(f"   Entry: ${signal.entry_price:.4f}")
        console.print(f"   Stop: ${signal.stop_loss:.4f}")
        console.print(f"   TP: ${signal.take_profit:.4f}")
        if signal.partial_take_profit:
            console.print(f"   Partial TP: ${signal.partial_take_profit:.4f} ({signal.partial_fraction:.0%})")
        console.print(f"   R:R = {signal.risk_reward_ratio:.2f}")
        console.print(f"   Strength: {signal.strength:.2f}")
        console.print(f"   Size: {size:.6f}")
        console.print(f"   Pair weight: {pair_weight:.2f}x | Strat weight: {self.risk.get_strategy_weight(signal.strategy):.2f}x")
        console.print(f"   Reason: {signal.reason}")

        self.logger.info(
            "SIGNAL %s %s %s str=%.2f size=%.6f SL=%.4f TP=%.4f | %s",
            signal.side.value, signal.symbol, signal.strategy,
            signal.strength, size, signal.stop_loss, signal.take_profit, signal.reason,
        )

        self.exchange.set_leverage(signal.symbol, signal.leverage)
        result = self.exchange.place_order(
            symbol=signal.symbol,
            side=signal.side,
            amount=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy,
            leverage=signal.leverage,
        )

        if "error" in result:
            console.print(f"[red]   Order failed: {result['error']}[/]")
            self.logger.error("Order failed: %s", result["error"])
            return

        # Attach partial TP metadata on paper positions
        if self.exchange.mode == "paper" and signal.symbol in self.exchange.paper_positions:
            pos = self.exchange.paper_positions[signal.symbol]
            pos.partial_take_profit = signal.partial_take_profit
            pos.partial_fraction = signal.partial_fraction
            pos.initial_size = size
            pos.strategy = signal.strategy

        # Live: stash partial levels in brackets
        if self.exchange.mode != "paper":
            br = self.exchange._local_brackets.get(signal.symbol, {})
            br["partial_take_profit"] = signal.partial_take_profit
            br["partial_fraction"] = signal.partial_fraction
            br["initial_size"] = size
            br["strategy"] = signal.strategy
            self.exchange._local_brackets[signal.symbol] = br

        order_id = str(result.get("id", "N/A"))
        console.print(f"[green]   Order filled: {order_id}[/]")
        self.logger.info(
            "FILL %s %s size=%.6f id=%s SL=%.4f TP=%.4f",
            signal.side.value, signal.symbol, size, order_id,
            signal.stop_loss, signal.take_profit,
        )

        # OrderId <-> decision linkage. WEEX requires the submitted AI logs to match
        # decisions to the orders they caused; reconstructing this after the fact is
        # guesswork, so bind it at the moment of the fill.
        if decision_id and self.decision_log:
            self.decision_log.link_order(
                decision_id,
                symbol=signal.symbol,
                order_id=order_id,
                side=signal.side.value,
                size=size,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            self.position_decisions[signal.symbol] = decision_id
        if result.get("sl_placed") is False and self.exchange.mode != "paper":
            console.print("[yellow]   SL not on exchange — software stop active[/]")
        if result.get("tp_placed") is False and self.exchange.mode != "paper":
            console.print("[yellow]   TP not on exchange — software TP active[/]")
        # Push open positions to dashboard immediately
        self._persist_state()

    def _manage_positions(self, account):
        candles_cache = {}

        for position in list(account.positions):
            try:
                ticker = self.exchange.fetch_ticker(position.symbol)
                if not ticker:
                    continue
                current_price = float(ticker.get("last") or position.entry_price)

                # Restore partial metadata from brackets if needed
                if self.exchange.mode != "paper":
                    br = self.exchange._local_brackets.get(position.symbol, {})
                    if not position.partial_take_profit and br.get("partial_take_profit"):
                        position.partial_take_profit = br["partial_take_profit"]
                        position.partial_fraction = br.get("partial_fraction", 0.5)
                        position.initial_size = br.get("initial_size") or position.size
                        position.partial_taken = br.get("partial_taken", False)

                if position.symbol not in candles_cache:
                    candles_cache[position.symbol] = self.exchange.fetch_candles(
                        position.symbol, "1h", 30
                    )
                candles = candles_cache[position.symbol]
                if len(candles) >= 14:
                    highs = np.array([c.high for c in candles])
                    lows = np.array([c.low for c in candles])
                    closes = np.array([c.close for c in candles])
                    atr = float(calculate_atr(highs, lows, closes)[-1])
                else:
                    atr = current_price * 0.015

                # Partial take-profit. apply_partial_tp() mutates the position, so
                # remember what to roll back to if the venue rejects the reduction.
                stop_before = position.stop_loss
                position, realized, closed = self.risk.apply_partial_tp(
                    position, current_price, atr
                )
                if realized is not None and closed > 0:
                    fee = closed * current_price * self.exchange.commission_rate
                    net = realized - fee
                    scaled_out = True

                    if self.exchange.mode == "paper":
                        self.exchange.balance += net
                        if position.size <= 1e-12:
                            self.exchange.paper_positions.pop(position.symbol, None)
                        else:
                            self.exchange.paper_positions[position.symbol] = position
                    else:
                        try:
                            side = "sell" if position.side == Side.LONG else "buy"
                            self.exchange.exchange.create_order(
                                position.symbol, "market", side, closed,
                                params={"reduceOnly": True},
                            )
                        except Exception as e:
                            # The exchange still holds the full position. Booking the
                            # PnL now would leave the bot managing a phantom size and
                            # a stop it never actually earned.
                            self.logger.error(
                                "Partial close failed, rolling back: %s", e
                            )
                            console.print(f"[red]Partial close failed: {e}[/]")
                            position.size += closed
                            position.partial_taken = False
                            position.stop_loss = stop_before
                            scaled_out = False

                    if scaled_out:
                        position.realized_pnl += net
                        position.fees_paid += fee
                        # Cash is banked now, so the daily loss limit must see it now.
                        self.risk.record_partial(net)
                        console.print(
                            f"[green]Partial TP: {position.symbol} closed {closed:.6f} "
                            f"PnL=${net:.2f} - stop to BE[/]"
                        )
                        self.logger.info(
                            "PARTIAL_TP %s closed=%.6f pnl=%.2f fee=%.4f remaining=%.6f",
                            position.symbol, closed, net, fee, position.size,
                        )
                        if position.size <= 1e-12:
                            continue

                position = self.risk.adjust_stops(position, current_price, atr)
                self.exchange.update_local_brackets(position)

                if self.exchange.mode == "paper" and position.symbol in self.exchange.paper_positions:
                    self.exchange.paper_positions[position.symbol] = position

                if position.should_stop_loss(current_price):
                    # A stop that has been trailed past entry is a protected exit,
                    # not a loss — labelling both "stop_loss" poisons exit analysis.
                    reason = position.stop_exit_reason()
                    color = "yellow" if reason == "be_stop" else "red"
                    console.print(f"[{color}]{reason}: {position.symbol}[/]")
                    self._close_position(position, current_price, reason)
                elif position.should_take_profit(current_price):
                    console.print(f"[green]Take-profit: {position.symbol}[/]")
                    self._close_position(position, current_price, "take_profit")
                elif position.should_trailing_stop(current_price):
                    console.print(f"[yellow]Trailing stop: {position.symbol}[/]")
                    self._close_position(position, current_price, "trailing_stop")

            except Exception as e:
                console.print(f"[red]Error managing {position.symbol}: {e}[/]")
                self.logger.exception("Manage %s: %s", position.symbol, e)

    def _close_position(self, position: Position, current_price: float, reason: str):
        result = self.exchange.close_position(position.symbol)
        if isinstance(result, dict) and result.get("error"):
            console.print(f"[red]   Close failed: {result['error']}[/]")
            return

        exit_price = (
            float(result.get("exit_price") or current_price)
            if isinstance(result, dict)
            else current_price
        )
        # Paper reports the net figure it actually booked; live has no per-trade
        # PnL in the close order, so charge the exit fee at the configured rate.
        if isinstance(result, dict) and result.get("pnl") is not None:
            final_leg = float(result["pnl"])
            exit_fee = float(result.get("fee") or 0.0)
        else:
            exit_fee = position.size * exit_price * self.exchange.commission_rate
            final_leg = position.calculate_pnl(exit_price) - exit_fee

        # Round trip = partial legs already banked + this leg, net of every fee
        # including entry. Win/loss and Kelly are driven off this number, so a
        # trade that only cleared the spread must not read as a winner.
        pnl = final_leg + position.realized_pnl - position.entry_fee
        fees = position.fees_paid + exit_fee

        # Margin is measured on the position as originally opened, otherwise a
        # scaled-out trade reports an inflated return on a shrunken base.
        sized_for_margin = position.initial_size or position.size
        notional_margin = sized_for_margin * position.entry_price / max(position.leverage, 1)
        pnl_pct = (pnl / notional_margin) * 100 if notional_margin else 0
        duration = int((datetime.utcnow() - position.opened_at).total_seconds())

        trade_result = TradeResult(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=sized_for_margin,
            leverage=position.leverage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_seconds=duration,
            exit_reason=reason,
            strategy=position.strategy,
            banked_pnl=position.realized_pnl,
            fees=fees,
        )
        self.risk.record_trade(trade_result)
        self.strategy.sync_scores_from_risk(self.risk)

        # Write the realised result back against the decision that opened it, so the
        # log shows not just what the model thought but what it actually earned.
        decision_id = self.position_decisions.pop(position.symbol, None)
        if decision_id and self.decision_log:
            self.decision_log.record_outcome(
                decision_id,
                symbol=position.symbol,
                order_id=str(result.get("id", "")) if isinstance(result, dict) else "",
                pnl=pnl,
                exit_price=exit_price,
                exit_reason=reason,
            )

        if pnl < 0:
            self.strategy.record_loss(datetime.utcnow())
        else:
            self.strategy.record_win()

        color = "green" if pnl >= 0 else "red"
        console.print(f"[{color}]   PnL: ${pnl:.2f} ({pnl_pct:.1f}%) — {reason}[/]")
        self.logger.info(
            "CLOSE %s %s pnl=%.2f banked=%.2f fees=%.4f reason=%s strategy=%s",
            position.symbol, position.side.value, pnl,
            position.realized_pnl, fees, reason, position.strategy,
        )
        self._persist_state()

    def _persist_state(self):
        try:
            lt = {
                k: (v.isoformat() if hasattr(v, "isoformat") else str(v))
                for k, v in self.strategy.last_trade_time.items()
            }
            account = self.exchange.snapshot_for_dashboard()
            # Keep a rolling equity tick for the dashboard curve (live mark-to-market)
            prev = load_state(self.state_path)
            ticks = list(prev.get("equity_ticks") or [])
            ticks.append({
                "t": datetime.utcnow().isoformat() + "Z",
                "equity": account.get("equity"),
                "balance": account.get("balance"),
                "unrealized": account.get("unrealized_pnl"),
                "open": account.get("open_positions"),
            })
            ticks = self._compact_ticks(ticks)

            save_state(
                self.state_path,
                {
                    "risk": self.risk.to_state(),
                    "paper": self.exchange.to_state(),
                    "account": account,
                    "equity_ticks": ticks,
                    "last_trade_time": lt,
                    # Survives restart so an open position keeps its provenance.
                    "position_decisions": self.position_decisions,
                    "cycle_count": self.cycle_count,
                    "bot_version": "v8.5",
                    "mode": self.config.get("trading", {}).get("mode", "paper"),
                },
            )
        except Exception as e:
            self.logger.warning("State save failed: %s", e)

    # Fine detail for the recent window, thinned history behind it. A flat
    # last-N cap at 60s/tick could only ever show ~8h, so a multi-day
    # competition curve lost its own trades off the left edge.
    RECENT_WINDOW_MINUTES = 120
    OLD_BUCKET_MINUTES = 15
    MAX_TICKS = 2000

    @classmethod
    def _compact_ticks(cls, ticks: list[dict]) -> list[dict]:
        if len(ticks) <= 2:
            return ticks

        def parsed(tick):
            try:
                return datetime.fromisoformat(str(tick.get("t", "")).replace("Z", ""))
            except Exception:
                return None

        now = datetime.utcnow()
        recent, buckets = [], {}
        for tick in ticks:
            t = parsed(tick)
            if t is None:
                continue
            if (now - t).total_seconds() <= cls.RECENT_WINDOW_MINUTES * 60:
                recent.append(tick)
            else:
                # Last tick in each bucket wins, so an exit is never averaged away.
                key = int(t.timestamp() // (cls.OLD_BUCKET_MINUTES * 60))
                buckets[key] = tick

        compacted = [buckets[k] for k in sorted(buckets)] + recent
        return compacted[-cls.MAX_TICKS:]

    def _display_status(self):
        if self.cycle_count % 5 != 0:
            return

        account = self.exchange.get_account_state()
        stats = self.risk.get_stats()

        table = Table(title=f"Status (Cycle #{self.cycle_count})", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Balance", f"${account.balance:.2f}")
        table.add_row("Equity", f"${account.equity:.2f}")
        table.add_row("Unrealized PnL", f"${account.unrealized_pnl:.2f}")
        table.add_row("Open Positions", str(len(account.positions)))
        table.add_row("Total Trades", str(stats.get("total_trades", 0)))
        table.add_row("Win Rate", f"{stats.get('win_rate', 0):.1%}")
        table.add_row("Total PnL", f"${stats.get('total_pnl', 0):.2f}")
        table.add_row("Sharpe", f"{stats.get('sharpe_ratio', 0):.2f}")
        table.add_row("Consec Losses", str(stats.get("consecutive_losses", 0)))

        pair_stats = stats.get("pair_stats", {})
        if pair_stats:
            table.add_row("", "")
            table.add_row("[bold]Pairs[/]", "")
            for symbol, ps in pair_stats.items():
                name = symbol.split("/")[0]
                table.add_row(
                    f"  {name}",
                    f"PnL=${ps['total_pnl']:.0f} | W={ps['weight']:.2f}x | n={ps['trades']}",
                )

        strat = stats.get("strategy_stats", {})
        if strat:
            table.add_row("", "")
            table.add_row("[bold]Strategies[/]", "")
            for name, ss in strat.items():
                table.add_row(
                    f"  {name}",
                    f"PnL=${ss['total_pnl']:.0f} | W={ss['weight']:.2f}x | n={ss['trades']}",
                )

        console.print(table)

    def _load_config(self, path: str) -> dict:
        config_path = Path(path)
        if not config_path.exists():
            return {}
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    def _shutdown(self, signum, frame):
        console.print("\n[yellow]Shutting down gracefully...[/]")
        self.running = False

    def _cleanup(self):
        stats = self.risk.get_stats()
        console.print(Panel.fit(
            f"[bold]Final Statistics[/]\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1%}\n"
            f"Total PnL: ${stats.get('total_pnl', 0):.2f}\n"
            f"Sharpe: {stats.get('sharpe_ratio', 0):.2f}",
            title="Session Ended",
        ))
        self.logger.info("Session ended stats=%s", stats)


def main():
    console.print("[bold green]WEEX AI Wars II — Trading Bot v8.5[/]")
    console.print("[dim]Press Ctrl+C to stop[/]\n")
    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()
