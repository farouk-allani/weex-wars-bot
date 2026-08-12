"""WEEX AI Wars II — Main Trading Engine v8.4

- HTF data, adaptive strategy scores
- State persistence across restarts
- Partial take-profit handling
- File logging
"""

import json
import logging
import os
import time
import urllib.request
import yaml
import signal as sig
from datetime import datetime, timedelta
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
from ..ai.client import TERMINAL_ERROR_KINDS
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
        # Entry conviction per open symbol. Needed to price a swap: replacing a
        # position is only rational if the replacement is genuinely better, and
        # without the incumbent's conviction there is nothing to compare against.
        self.position_conviction: dict[str, float] = {
            str(k): float(v)
            for k, v in ((state or {}).get("position_conviction") or {}).items()
        }
        self._last_ai_call: datetime | None = None
        self.ai_interval_min = float(ai_cfg.get("decision_interval_minutes", 60))
        # Oscillators + edge_signals travel together, exactly like the replay's
        # --no-osc arm. Off unless someone re-tests and flips it in config: feeding
        # them triggered a memorised BB-fade reflex that shorted a bull market.
        self.ai_include_osc = bool(ai_cfg.get("include_oscillators", False))

        # --- Maker execution ---
        # Entries rest at the touch instead of crossing the spread: measured
        # round-trip cost at market (0.22%) exceeded the best measured edge
        # (~0.13%/trade), so the fee/slippage saved here IS the strategy's margin.
        exec_cfg = self.config.get("execution", {}) or {}
        self.maker_entries = bool(exec_cfg.get("maker_entries", False))
        self.entry_ttl_sec = float(exec_cfg.get("entry_ttl_minutes", 45)) * 60
        self.reprice_sec = float(exec_cfg.get("reprice_seconds", 180))
        self.max_chase_atr = float(exec_cfg.get("max_chase_atr", 0.5))
        # symbol -> resting entry bookkeeping (order id, brackets, chase state).
        # Restored across restarts; the first _manage_pending_entries pass
        # reconciles each entry against the venue/paper ledger and drops the dead.
        self.pending_entries: dict[str, dict] = (state or {}).get("pending_entries") or {}

        # AI liveness. A failed decision call fails closed (no decision = hold), which
        # is correct but indistinguishable from a calm market — so the failures are
        # counted and surfaced. Without this, a retired model id bought us 16h of a
        # bot that logged cheerfully and did not think.
        self.ai_max_failures = int(ai_cfg.get("max_consecutive_failures", 3))
        self._ai_consecutive_failures = 0
        self._last_ai_success: datetime | None = None
        self.ai_health_path = self.state_path.parent / "ai_health.json"
        # Provider credit. Measured 2026-08-12: the account hit zero on 08-07 and
        # the bot sat blind for 65 hours — 39% of a weekly round — because nothing
        # watched the one dependency that expires on a clock nobody reads. The
        # model-id preflight could not see it: /models answers fine at zero balance.
        self.ai_balance_warn_usd = float(ai_cfg.get("balance_warn_usd", 2.0))
        self.ai_balance_check_hours = float(ai_cfg.get("balance_check_hours", 6))
        self._ai_balance: dict | None = None
        self._last_balance_check: datetime | None = None
        # Alarm de-duplication: alert on the EDGE, not once per cycle. A brain-dead
        # bot that pages every hour for three days trains you to ignore the page.
        self._alerted_state: str | None = None
        # ai-log coverage, published next to it. Alarms on change, not every cycle.
        self.compliance_path = self.state_path.parent / "compliance.json"
        self._last_compliance_gap = -1

        if ai_cfg.get("enabled", False):
            self.decision_log = DecisionLog(ai_cfg.get("log_file", "logs/ai_decisions.jsonl"))
            client = DeepSeekClient(self.config)
            # Fail at startup on a dead model id rather than once an hour, silently,
            # forever. A listing outage is not fatal — the failure alarm covers that.
            try:
                client.validate_model()
                self.logger.info("AI model %s validated against provider", client.model)
            except Exception as e:
                self.logger.error("AI MODEL VALIDATION FAILED: %s", e)
                console.print(f"[bold red]AI model validation failed: {e}[/]")
            # The second half of the preflight: is there money behind the key? Free
            # to ask, and it is the check that the 65h outage needed.
            self._check_ai_balance(client, force=True)
            self.ai = AITrader(self.config, client, self.decision_log)
            self.logger.info(
                "AI decision layer active: model=%s interval=%smin alarm_after=%d failures",
                self.ai.client.model, self.ai_interval_min, self.ai_max_failures,
            )
            # Publish liveness immediately, before the first hourly decision. The
            # healthcheck must be able to treat a MISSING file as unhealthy — but
            # it can only do that if the file is guaranteed to exist once the
            # process is up. Otherwise "no file" means both "booting normally" and
            # "the writer never ran", and the check has to accept the file's
            # absence, which is exactly the hole that let a dead decision layer
            # report healthy.
            self._record_ai_health(None, decided=False)

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
        # Resting entries are managed every cycle (60s), same as stops — a fill,
        # a chase or an abandon must never wait on the model's hourly cadence.
        self._manage_pending_entries()
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
        # Closes are kept so the correlation matrix comes free off data we already
        # fetch — the portfolio risk budget needs covariance, not a position count.
        closes_by_symbol: dict[str, np.ndarray] = {}
        for symbol in symbols:
            try:
                candles = self.exchange.fetch_candles(symbol, timeframe, lookback)
                if len(candles) < 100:
                    # Never silent: a symbol missing here is invisible to the model,
                    # and if we hold it, the model reasons about holding or closing
                    # it with no tape in front of it.
                    held = any(p.symbol == symbol for p in account.positions)
                    self.logger.warning(
                        "AI context: %s dropped, only %d candles (held=%s)",
                        symbol, len(candles), held,
                    )
                    continue
                htf_candles = self.exchange.fetch_candles(symbol, htf, htf_lookback)
                funding = self.exchange.fetch_funding_rate(symbol)

                highs = np.array([c.high for c in candles])
                lows = np.array([c.low for c in candles])
                closes = np.array([c.close for c in candles])
                atrs[symbol] = float(calculate_atr(highs, lows, closes)[-1])
                prices[symbol] = float(closes[-1])
                closes_by_symbol[symbol] = closes

                edges = (
                    self.edges.analyze_all_edges(
                        candles, funding, higher_tf_candles=htf_candles or None
                    )
                    if self.ai_include_osc
                    else None
                )
                market.append(
                    symbol_snapshot(
                        symbol, candles, funding, htf_candles or None, edges,
                        include_oscillators=self.ai_include_osc,
                    )
                )
            except Exception as e:
                self.logger.exception("AI context build failed for %s: %s", symbol, e)

        if not market:
            self.logger.warning("AI cycle skipped: no market data")
            return

        correlations = self.risk.returns_correlation(
            closes_by_symbol, self.risk.correlation_lookback
        )

        context = build_context(
            symbols_data=market,
            account=account,
            risk=self.risk,
            recent_trades=self.risk.trade_history,
            competition=self._competition_context(),
            position_conviction=self.position_conviction,
        )
        # Tell the model plainly when it may not open anything, so it reasons about
        # exits and holds instead of proposing entries that will be thrown away.
        if not can_trade:
            context["hard_limits"]["entries_blocked"] = reason
        self._add_correlation_context(context, account, correlations)

        console.print(f"[cyan]Consulting {self.ai.client.model}...[/]")
        decisions, assessment, decision_id = self.ai.decide(context)
        self._last_ai_call = datetime.utcnow()
        self._record_ai_health(self.ai.last_error)
        self._record_compliance()
        # Cheap, throttled, and on the success path too: the whole point is to see
        # the balance falling while the bot is still working.
        self._check_ai_balance()

        if self.ai.last_error:
            # Deliberately ERROR, not INFO. The previous outage was invisible because
            # a dead brain logged an empty assessment at the same level as a quiet
            # one. Execution below still runs: with no decisions it opens and closes
            # nothing, but the risk-block safety that pulls resting orders must not
            # be skipped just because the model is unreachable.
            self.logger.error(
                "AI CALL FAILED (%d consecutive, last success %s): %s",
                self._ai_consecutive_failures,
                self._last_ai_success.isoformat() if self._last_ai_success else "never",
                self.ai.last_error,
            )
            console.print(f"[bold red]AI call failed: {self.ai.last_error}[/]")
        else:
            if assessment:
                console.print(f"[dim]{assessment}[/]")
            self.logger.info("AI decision_id=%s assessment=%s", decision_id, assessment[:200])

        allowed = set(symbols)
        # Exits first: closing frees a slot that an entry below may want. That
        # ordering is what makes a swap possible at all, and it is also how
        # capacity pressure leaks into the exit decision — so the swap has to be
        # priced before any close is executed.
        blocked_swaps = self._capacity_only_closes(decisions, account)
        for d in decisions:
            if str(d.get("action", "")).lower() != "close":
                continue
            sym = d.get("symbol")
            pos = next((p for p in account.positions if p.symbol == sym), None)
            if not pos:
                continue
            if sym in blocked_swaps:
                console.print(f"[yellow]AI close {sym} refused: {blocked_swaps[sym]}[/]")
                self.logger.info("AI_CLOSE_REFUSED %s: %s", sym, blocked_swaps[sym])
                continue
            price = prices.get(sym) or pos.entry_price
            console.print(f"[yellow]AI closing {sym}: {d.get('rationale', '')[:100]}[/]")
            self.logger.info("AI_CLOSE %s reason=%s", sym, str(d.get("rationale"))[:200])
            self._close_position(pos, price, "ai_close")

        # A fresh decision supersedes a resting order that no longer matches it —
        # the model has seen an hour of data the order has not. Same-side entries
        # keep resting (the chase logic owns the price); everything else is pulled.
        for d in decisions:
            sym = str(d.get("symbol") or "")
            pe = self.pending_entries.get(sym)
            if not pe:
                continue
            act = str(d.get("action", "")).lower()
            if act in ("long", "short") and act == pe["side"]:
                continue
            self._abandon_pending(sym, pe, f"superseded_by_{act or 'none'}")

        if not can_trade:
            console.print(f"[yellow]Entries blocked: {reason}[/]")
            self.logger.info("AI entries blocked: %s", reason)
            # Resting orders are entries too. A block that leaves them on the book
            # would keep opening positions the risk engine just refused.
            for sym, pe in list(self.pending_entries.items()):
                self._abandon_pending(sym, pe, f"blocked:{reason}")
            return

        account = self.exchange.get_account_state()
        for d in decisions:
            act = str(d.get("action", "")).lower()
            if act not in ("long", "short"):
                continue
            sym = str(d.get("symbol") or "")
            if sym not in prices:
                continue
            if any(p.symbol == sym for p in account.positions):
                continue
            if sym in self.pending_entries:
                continue  # same thesis already resting; chase logic owns it
            # Resting orders are commitments: count them against the caps, or a
            # cycle of unfilled entries quietly stacks exposure past the limits.
            if len(account.positions) + len(self.pending_entries) >= self.risk.max_open_positions:
                break
            pending_same = sum(1 for q in self.pending_entries.values() if q["side"] == act)
            open_same = sum(1 for p in account.positions if p.side.value == act)
            if 0 < self.risk.max_same_side_positions <= pending_same + open_same:
                self.logger.info(
                    "AI_VETO %s: %d %s position(s)/order(s) already committed",
                    sym, pending_same + open_same, act,
                )
                continue

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

            # Correlation-aware portfolio budget. Scales rather than vetoes: a
            # second, highly-correlated leg is not forbidden, it just cannot be
            # full size, because the risk that matters is what happens when every
            # stop fills in the same move.
            scale, why_scale = self.risk.correlation_scale(
                signal, size, account, correlations,
                pending=self._pending_as_legs(),
            )
            if scale <= 0:
                console.print(f"[yellow]AI {sym} vetoed: {why_scale}[/]")
                self.logger.info("AI_VETO %s: %s", sym, why_scale)
                continue
            if scale < 1.0:
                size *= scale
                console.print(f"[yellow]   {sym} {why_scale}[/]")
                self.logger.info("RISK_SCALE %s size x%.3f — %s", sym, scale, why_scale)
                if size * signal.entry_price < self.risk.min_position_usd:
                    self.logger.info(
                        "AI_VETO %s: scaled size below minimum notional", sym
                    )
                    continue

            self._execute_trade(
                signal, size, pair_weight,
                decision_id=decision_id, atr=atrs.get(sym, 0.0),
            )
            account = self.exchange.get_account_state()

    def _capacity_only_closes(self, decisions: list, account) -> dict[str, str]:
        """Closes that exist only to free a slot, mapped to why they were refused.

        Measured 2026-08-01: with the book full, the model closed a losing SOL short
        with the stated reason "Closing the short to free a position slot", then
        re-opened the same short 62 minutes later — a realised -3.03 and two extra
        round trips to arrive back at the position it already had. The prompt had
        explicitly sanctioned this ("the margin slot is needed for a clearly better
        opportunity"), but nothing ever checked the "clearly better" part.

        So check it. A swap is legitimate when the replacement genuinely beats the
        incumbent; below that margin the close is capacity pressure wearing a
        thesis. Closes with no replacement competing for their slot are untouched —
        those are ordinary thesis exits, and this guard has no opinion on them.
        """
        margin = self.risk.swap_conviction_margin
        if margin <= 0:
            return {}

        committed = len(account.positions or []) + len(self.pending_entries)
        if committed < self.risk.max_open_positions:
            return {}  # a free slot exists; no close is buying one

        open_syms = {p.symbol for p in (account.positions or [])}
        candidates = []
        for d in decisions:
            if str(d.get("action", "")).lower() not in ("long", "short"):
                continue
            sym = str(d.get("symbol") or "")
            if not sym or sym in open_syms or sym in self.pending_entries:
                continue  # not asking for a new slot
            try:
                candidates.append((float(d.get("conviction") or 0.0), sym))
            except (TypeError, ValueError):
                continue
        if not candidates:
            return {}  # nothing wants the slot, so nothing is being freed for it

        closes = []
        for d in decisions:
            if str(d.get("action", "")).lower() != "close":
                continue
            sym = str(d.get("symbol") or "")
            if sym in open_syms:
                closes.append((self._entry_conviction(sym), sym))
        if not closes:
            return {}

        # Pair the cheapest incumbents against the strongest replacements: that is
        # the most favourable reading of the model's intent, so anything this
        # pairing still rejects is churn under any reading.
        candidates.sort(reverse=True)
        closes.sort()
        refused: dict[str, str] = {}
        for (held_conv, held_sym), (new_conv, new_sym) in zip(closes, candidates):
            if new_conv < held_conv + margin:
                refused[held_sym] = (
                    f"capacity swap for {new_sym} at conviction {new_conv:.2f} does not "
                    f"beat held conviction {held_conv:.2f} by {margin:.2f} — holding"
                )
        return refused

    def _entry_conviction(self, symbol: str) -> float:
        """Conviction this position was opened at.

        Unknown for anything opened before this was tracked, and for those the
        honest fallback is the lowest conviction that could have opened it — assume
        the weakest legal incumbent rather than block a swap on missing data.
        """
        known = self.position_conviction.get(symbol)
        if known is not None:
            return float(known)
        return float(getattr(self.ai, "min_conviction", 0.0) or 0.0)

    def _ai_due(self) -> bool:
        if self._last_ai_call is None:
            return True
        elapsed_min = (datetime.utcnow() - self._last_ai_call).total_seconds() / 60
        return elapsed_min >= self.ai_interval_min

    def _add_correlation_context(self, context: dict, account, correlations: dict):
        """Show the model its correlation exposure, not just its position count.

        Without this the budget is invisible to the decision-maker: it proposes a
        second correlated leg at full conviction, the engine silently halves it, and
        the model never learns that a diversifying trade would have been worth more.
        """
        try:
            # Resting entries are included for the same reason they are included in
            # the budget itself: understating what is already committed would show
            # the model free capacity that the risk engine is about to refuse, and
            # this context exists precisely so the two agree.
            positions = list(account.positions or []) + self._pending_as_legs()
            eq = account.equity or 0.0
            used = self.risk.portfolio_risk(positions, correlations)
            used_n = self.risk.correlated_notional(positions, correlations)
            hl = context.setdefault("hard_limits", {})

            # Budget 1: tail loss if every stop fires together.
            hl["max_portfolio_stopout_risk_pct"] = round(
                self.risk.max_portfolio_risk * 100, 2
            )
            hl["portfolio_stopout_risk_used_pct"] = (
                round(used / eq * 100, 3) if eq else 0.0
            )
            # Budget 2: how hard the equity curve can swing. Usually the binding one.
            hl["max_correlated_notional_x_equity"] = round(
                self.risk.max_correlated_notional, 2
            )
            hl["correlated_notional_used_x_equity"] = (
                round(used_n / eq, 3) if eq else 0.0
            )
            hl["portfolio_risk_note"] = (
                "Two correlation budgets, both scored. The stop-out one is what you "
                "lose if EVERY open stop fills in the same move. The notional one is "
                "how hard your equity curve can swing, which is what the stability "
                "metric sees. Same-side positions in correlated pairs consume both "
                "nearly additively, so a leg that doubles an existing bet is "
                "automatically scaled down, while one that diversifies or hedges is "
                "not. Prefer the diversifying trade when setups are comparable."
            )

            # The book's own correlations, so "diversifying" is a fact not a guess.
            if correlations and account.positions:
                held = [p.symbol for p in account.positions]
                pairs = {}
                for (a, b), c in correlations.items():
                    if a in held or b in held:
                        pairs[f"{a.split('/')[0]}~{b.split('/')[0]}"] = round(c, 2)
                if pairs:
                    top = dict(
                        sorted(pairs.items(), key=lambda kv: -abs(kv[1]))[:12]
                    )
                    hl["correlations_vs_open_book"] = top
            exhausted = [
                name
                for name, u, cap in (
                    ("stop-out risk", used, self.risk.max_portfolio_risk * eq),
                    ("correlated notional", used_n, self.risk.max_correlated_notional * eq),
                )
                if cap > 0 and u >= cap
            ]
            if exhausted:
                hl["portfolio_risk_exhausted"] = (
                    f"{' and '.join(exhausted)} budget is full — new entries will be "
                    "rejected until a position closes or a stop tightens. Reason about "
                    "exits and holds, not entries."
                )
        except Exception as e:  # context enrichment must never break a cycle
            self.logger.warning("correlation context failed: %s", e)

    def _check_ai_balance(self, client=None, *, force: bool = False) -> dict | None:
        """Poll remaining provider credit, at most every `balance_check_hours`.

        Deliberately cheap and deliberately separate from the decision call: the
        point is to see the wall BEFORE we hit it, and a probe that only runs when
        a decision fails is just a slower way of finding out too late.
        """
        client = client or (self.ai.client if self.ai else None)
        if client is None or not hasattr(client, "check_balance"):
            return None

        now = datetime.utcnow()
        if not force and self._last_balance_check is not None:
            elapsed_h = (now - self._last_balance_check).total_seconds() / 3600.0
            if elapsed_h < self.ai_balance_check_hours:
                return self._ai_balance

        result = client.check_balance()
        self._last_balance_check = now
        if result.get("error"):
            # An unreachable balance endpoint is not evidence of an empty account;
            # keep the last known reading rather than overwriting it with a blank.
            self.logger.warning("AI balance check failed: %s", result["error"])
            return self._ai_balance

        self._ai_balance = result
        bal, avail = result.get("balance"), result.get("available")
        if avail is False or (bal is not None and bal <= 0):
            self._alert(
                "ai_credit_exhausted",
                f"DeepSeek credit is EXHAUSTED (balance {bal} {result.get('currency') or ''}). "
                "The bot cannot think and will not open a position until it is topped up.",
            )
        elif bal is not None and bal <= self.ai_balance_warn_usd:
            self._alert(
                "ai_credit_low",
                f"DeepSeek credit is LOW: {bal} {result.get('currency') or ''} left "
                f"(warn threshold {self.ai_balance_warn_usd}). "
                "Top up before the next round — a mid-round outage costs the trade minimum.",
            )
        else:
            self._clear_alert("ai_credit_low")
            self._clear_alert("ai_credit_exhausted")
            self.logger.info(
                "AI balance OK: %s %s", bal, result.get("currency") or "",
            )
        return result

    def _alert(self, key: str, message: str):
        """Raise an operational alarm once per state transition.

        Writes to `data/alerts.json` (the dashboard reads it) and, if
        `ALERT_WEBHOOK_URL` is set, POSTs there. The webhook is opt-in because the
        repo must not carry a secret, and it is best-effort because an alerting
        outage must never stop the trading loop.
        """
        if self._alerted_state == key:
            return
        self._alerted_state = key
        self.logger.critical("ALERT[%s] %s", key, message)
        console.print(Panel.fit(f"[bold red]{message}[/]", title=f"ALERT: {key}"))

        payload = {
            "key": key,
            "message": message,
            "at": datetime.utcnow().isoformat() + "Z",
            "bot": self.config.get("trading", {}).get("mode", "paper"),
        }
        try:
            path = self.state_path.parent / "alerts.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            history = []
            if path.exists():
                history = json.loads(path.read_text() or "[]")
            history.append(payload)
            path.write_text(json.dumps(history[-50:], indent=1))
        except Exception as e:
            self.logger.warning("could not write alerts.json: %s", e)

        url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        if not url:
            return
        try:
            body = json.dumps({"text": f"[weex-bot] {message}", **payload}).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as e:
            self.logger.warning("alert webhook failed: %s", e)

    def _clear_alert(self, key: str):
        if self._alerted_state == key:
            self._alerted_state = None

    def _record_ai_health(self, error: str | None, *, decided: bool = True):
        """Publish decision-layer liveness so something outside the process can see it.

        `bot_state.json` keeps being written on every cycle whether or not the model
        answers, so its mtime cannot distinguish a thinking bot from a brain-dead
        one — which is exactly how a 16h outage passed every healthcheck. This file
        is the signal the container healthcheck actually keys on.

        `decided=False` publishes the file at boot without claiming a decision
        happened: no counter moves and `last_success` stays None. Stamping boot time
        into a success field would be the same lie that once put boot timestamps on
        restored trades — the file exists to say "the writer is alive", and it must
        not answer "when did the model last work?" with a time it did not work.
        """
        if decided:
            if error:
                self._ai_consecutive_failures += 1
            else:
                self._ai_consecutive_failures = 0
                self._last_ai_success = datetime.utcnow()

        kind = getattr(self.ai, "last_error_kind", None) if self.ai else None
        # A billing wall, a revoked key or a retired model is terminal on the FIRST
        # occurrence — waiting for three of them buys nothing but two more hours of
        # blindness, and the counter was the reason the 65h outage stayed quiet.
        terminal = bool(error) and kind in TERMINAL_ERROR_KINDS
        healthy = (not terminal) and self._ai_consecutive_failures < self.ai_max_failures
        try:
            self.ai_health_path.parent.mkdir(parents=True, exist_ok=True)
            self.ai_health_path.write_text(json.dumps({
                "healthy": healthy,
                "consecutive_failures": self._ai_consecutive_failures,
                "max_consecutive_failures": self.ai_max_failures,
                "last_success": (
                    self._last_ai_success.isoformat() + "Z" if self._last_ai_success else None
                ),
                "last_error": error,
                "last_error_kind": kind,
                "terminal": terminal,
                "balance": self._ai_balance,
                "model": self.ai.client.model if self.ai else None,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }, indent=1))
        except Exception as e:  # never let telemetry break the trading loop
            self.logger.warning("could not write ai_health: %s", e)

        if terminal:
            # Re-probe the balance so the alert carries the actual number rather
            # than the reader having to go and look it up.
            self._check_ai_balance(force=True)
            self._alert(
                f"ai_{kind}",
                f"AI DECISION LAYER DOWN ({kind}): {error}. "
                + {
                    "billing": "Top up the DeepSeek account — the bot is blind until you do.",
                    "auth": "The API key is rejected. Check DEEPSEEK_API_KEY in .env on the VPS.",
                    "model": "The configured model id is gone. Update ai.model in config.yaml.",
                }.get(kind, "Human intervention required."),
            )
        elif not healthy:
            self.logger.critical(
                "AI DECISION LAYER DOWN: %d consecutive failures (>= %d). "
                "The bot is holding everything and cannot trade. Last error: %s",
                self._ai_consecutive_failures, self.ai_max_failures, error,
            )
            console.print(Panel.fit(
                f"[bold red]AI DECISION LAYER DOWN[/]\n"
                f"{self._ai_consecutive_failures} consecutive failures\n"
                f"{error}",
                title="ALARM",
            ))
        elif not error:
            self._clear_alert("ai_billing")
            self._clear_alert("ai_auth")
            self._clear_alert("ai_model")

    def _record_compliance(self):
        """Publish ai-log coverage: every linked order must have an ai-log file.

        An AI-driven live order without its ai-log is non-compliant, and the penalty
        is not a worse score but potentially no score — so coverage is monitored on
        every cycle rather than checked by hand before a round. Derived from the
        artifacts on disk, so it reports what a reviewer would see.
        """
        if not self.decision_log:
            return
        try:
            status = self.decision_log.compliance_status()
            status["updated_at"] = datetime.utcnow().isoformat() + "Z"
            self.compliance_path.parent.mkdir(parents=True, exist_ok=True)
            self.compliance_path.write_text(json.dumps(status, indent=1, default=str))
        except Exception as e:
            self.logger.warning("could not write compliance status: %s", e)
            return

        # Alarm on what is actionable: a missing log, or one that is incomplete
        # despite the record holding everything needed to complete it. Logs that
        # provably cannot be repaired are counted separately and stay quiet, so
        # they cannot desensitise us to a new failure.
        gap = int(status.get("orders_without_ai_log") or 0) + int(
            status.get("ai_logs_repairable_incomplete") or 0
        )
        if gap and gap != self._last_compliance_gap:
            self.logger.critical(
                "COMPLIANCE: %d of %d linked orders lack a usable ai-log "
                "(%d missing, %d incomplete-but-repairable). AI-driven orders "
                "without a valid log are non-compliant. Last error: %s",
                gap, status.get("orders_linked"),
                status.get("orders_without_ai_log"),
                status.get("ai_logs_repairable_incomplete"),
                status.get("last_error"),
            )
            console.print(Panel.fit(
                f"[bold red]{gap} ORDER(S) WITHOUT A USABLE AI-LOG[/]\n"
                f"{status.get('orders_linked')} orders linked, "
                f"{status.get('ai_logs_on_disk')} logs on disk\n"
                f"run: python run_compliance_audit.py --backfill",
                title="COMPLIANCE ALARM",
            ))
        self._last_compliance_gap = gap

    def _competition_context(self) -> dict:
        """Scoring, trade count and clock — the model reasons about all three."""
        comp = self.config.get("competition", {}) or {}
        real = [t for t in self.risk.trade_history if not self.risk.is_keepalive(t.strategy)]
        stats = self.risk.get_stats()
        ctx = {
            # AI Wars II is scored on profit AND risk AND stability. Telling the
            # model "cumulative PnL" (the AI Wars I rule) pointed it at variance
            # while the scoring punishes exactly that.
            "scoring": "multi-metric: realised profit + risk management + strategy stability",
            "ranking_metric": (
                "NOT cumulative PnL alone. A smaller steady gain with shallow "
                "drawdown outranks a larger erratic one."
            ),
            "trades_executed": len(real),
            "minimum_trades_required": comp.get("min_trades", 10),
            # Let it see the metrics it is judged on, not just its P&L.
            "current_win_rate": round(stats.get("win_rate", 0.0), 3),
            "current_sharpe": round(stats.get("sharpe_ratio", 0.0), 2),
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
        ctx["pace"] = self._trade_pace(comp, real)
        return ctx

    def _trade_pace(self, comp: dict, real: list) -> dict:
        """How the trade count is tracking against the round's minimum.

        The minimum is per round, and rounds are short (weekly). At the observed
        ~0.9 trades/day a 7-day round lands near 6 trades — under a 10-trade floor.
        So the count has to be watched, but the previous answer (mechanical keepalive
        in-outs) is wrong twice over for AI Wars II: a code-generated order carries no
        model decision, so it would ship with no ai-log and be non-compliant, and
        robotic heartbeat trades are exactly the erratic behaviour the stability
        metric punishes.

        Instead the constraint is handed to the model as a fact about its situation.
        It already picks instrument, direction and levels; asking it to spend its
        remaining budget on the best marginal setups it can find beats having code
        pick a blind one, and every resulting order keeps its reasoning trail.
        """
        min_trades = int(comp.get("min_trades", 10) or 0)
        round_days = float(comp.get("round_days", 7) or 7)

        now = datetime.utcnow()
        started = comp.get("round_started")
        round_start = None
        if started:
            try:
                round_start = datetime.fromisoformat(str(started).replace("Z", ""))
            except Exception:
                round_start = None

        if round_start is None:
            # PRESEASON / unanchored: there is no round clock, so a countdown would be
            # fiction. Report the trailing window as a rehearsal rate instead — the
            # question that can honestly be answered is "would this pace qualify?".
            window_start = now - timedelta(days=round_days)
            recent = [t for t in real if t.timestamp and t.timestamp >= window_start]
            per_day = len(recent) / round_days
            projected = per_day * round_days
            return {
                "round_days": round_days,
                "round_anchored": False,
                "trades_last_%dd" % int(round_days): len(recent),
                "observed_trades_per_day": round(per_day, 2),
                "projected_trades_per_round": round(projected, 1),
                "minimum_trades_required": min_trades,
                "status": (
                    f"PRESEASON (no round clock). At the current {per_day:.2f} "
                    f"trades/day a {round_days:.0f}-day round would finish with about "
                    f"{projected:.0f} trades against a {min_trades}-trade minimum — "
                    + (
                        "that would QUALIFY. Optimise purely for quality."
                        if projected >= min_trades
                        else "that would MISS the minimum. Treat this as evidence that "
                        "your bar for 'tradeable' is too high for a weekly round, and "
                        "take good-but-imperfect setups you would currently pass on."
                    )
                ),
            }

        elapsed_h = max((now - round_start).total_seconds() / 3600, 0.1)
        remaining_h = max(round_days * 24 - elapsed_h, 0.0)
        in_round = [t for t in real if t.timestamp and t.timestamp >= round_start]
        still_needed = max(min_trades - len(in_round), 0)
        required_per_day = (
            round(still_needed / (remaining_h / 24), 2) if remaining_h > 1 else None
        )
        observed_per_day = round(len(in_round) / (elapsed_h / 24), 2)

        pace = {
            "round_days": round_days,
            "round_anchored": True,
            "trades_this_round": len(in_round),
            "trades_still_needed": still_needed,
            "hours_elapsed": round(elapsed_h, 1),
            "hours_remaining_in_round": round(remaining_h, 1),
            "observed_trades_per_day": observed_per_day,
            "required_trades_per_day": required_per_day,
        }

        # Only nag when the arithmetic actually says we are short, and say by how
        # much. A vague "trade more" instruction is how a bot starts churning.
        if still_needed == 0:
            pace["status"] = (
                f"minimum already met ({len(in_round)}/{min_trades}) — extra trades "
                "are now pure cost. Quality only."
            )
        elif remaining_h <= 0:
            pace["status"] = (
                f"round is over with {len(in_round)}/{min_trades} trades. Nothing "
                "further to do about the count."
            )
        elif required_per_day and required_per_day > observed_per_day:
            pace["status"] = (
                f"BEHIND PACE: {still_needed} more trades needed in "
                f"{remaining_h:.0f}h to clear the {min_trades}-trade minimum "
                f"({required_per_day}/day required vs {observed_per_day}/day so far). "
                "Widen what you consider tradeable — take your best available setups "
                "at moderate conviction rather than waiting for perfect ones — but do "
                "not breach risk limits and do not open a position you cannot justify."
            )
        else:
            pace["status"] = (
                f"on pace ({len(in_round)}/{min_trades} with {remaining_h:.0f}h left)"
            )
        return pace

    def _execute_trade(
        self,
        signal: Signal,
        size: float,
        pair_weight: float = 1.0,
        decision_id: str | None = None,
        atr: float = 0.0,
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

        if self.maker_entries:
            self._place_maker_entry(signal, size, atr=atr, decision_id=decision_id)
            return

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
        # Recorded regardless of decision linkage: the swap guard needs it even for
        # positions opened by a non-AI strategy.
        self.position_conviction[signal.symbol] = float(signal.strength)
        if result.get("sl_placed") is False and self.exchange.mode != "paper":
            console.print("[yellow]   SL not on exchange — software stop active[/]")
        if result.get("tp_placed") is False and self.exchange.mode != "paper":
            console.print("[yellow]   TP not on exchange — software TP active[/]")
        # Push open positions to dashboard immediately
        self._persist_state()

    # ---- Maker entry lifecycle ----
    #
    # place -> (chase toward the touch) -> fill | abandon. The invariants:
    #   - one resting order per symbol, counted against position limits;
    #   - brackets travel with the order, so a fill can never be stopless;
    #   - we reprice toward the market but never past max_chase_atr from the
    #     signal price, and never below the minimum R:R — a trade that must be
    #     chased that far has already told us the setup is gone;
    #   - a missed entry costs nothing. Crossing the spread costs 0.11% every
    #     time. At our measured edge, patience is the whole profit margin.

    def _place_maker_entry(
        self,
        signal: Signal,
        size: float,
        atr: float = 0.0,
        decision_id: str | None = None,
    ):
        touch = self.exchange.touch_price(signal.symbol, signal.side)
        if touch <= 0:
            self.logger.warning("ENTRY_SKIP %s: no touch price", signal.symbol)
            return
        # A limit already beyond the stop or target is not an entry, it's a mistake.
        if signal.side == Side.LONG and not (signal.stop_loss < touch < signal.take_profit):
            self.logger.info("ENTRY_SKIP %s: touch %.6f outside bracket", signal.symbol, touch)
            return
        if signal.side == Side.SHORT and not (signal.take_profit < touch < signal.stop_loss):
            self.logger.info("ENTRY_SKIP %s: touch %.6f outside bracket", signal.symbol, touch)
            return

        res = self.exchange.place_entry_limit(
            signal.symbol, signal.side, size, touch,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy,
            leverage=signal.leverage,
            partial_take_profit=signal.partial_take_profit,
            partial_fraction=signal.partial_fraction,
        )
        if "error" in res:
            # Post-only rejection means the touch moved through us between the
            # ticker read and the order landing. Not an error to retry blindly —
            # the next cycle re-decides with fresh prices.
            console.print(f"[yellow]   Maker entry not placed: {res['error']}[/]")
            self.logger.info("ENTRY_PLACE_FAILED %s: %s", signal.symbol, res["error"])
            return

        now_iso = datetime.utcnow().isoformat()
        self.pending_entries[signal.symbol] = {
            "order_id": str(res["id"]),
            "side": signal.side.value,
            "size": size,
            "limit_price": touch,
            "signal_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "partial_take_profit": signal.partial_take_profit,
            "partial_fraction": signal.partial_fraction,
            "strategy": signal.strategy,
            "leverage": signal.leverage,
            # Carried to the fill so a maker-entered position knows what it cost
            # in conviction — the swap guard prices replacements against it.
            "strength": float(signal.strength),
            # Chase distances are measured in ATR; when the caller has none
            # (rules path), infer it from the stop, which to_signal keeps at
            # 0.5-4 ATR — the midpoint makes dist/2 a serviceable stand-in.
            "atr": atr if atr > 0 else abs(signal.entry_price - signal.stop_loss) / 2,
            "min_rr": self.ai.min_rr if self.ai else float(
                self.config.get("competition", {}).get("min_rr", 1.35)
            ),
            "decision_id": decision_id,
            "placed_at": now_iso,
            "last_reprice_at": now_iso,
            "reprices": 0,
        }
        console.print(
            f"[cyan]   Entry resting at {touch:.6f} (maker) — "
            f"chases <= {self.max_chase_atr}x ATR, TTL {self.entry_ttl_sec / 60:.0f}min[/]"
        )
        self.logger.info(
            "ENTRY_RESTING %s %s size=%.6f limit=%.6f id=%s",
            signal.side.value, signal.symbol, size, touch, res["id"],
        )
        self._persist_state()

    def _pending_as_legs(self) -> list[Position]:
        """Resting entries as position-shaped legs for the correlation budgets.

        Priced at the limit, because that is where they fill if they fill at all.
        Everything the budgets read — symbol, side, entry, size, stop — is already
        carried on the pending order; the rest of the Position is scaffolding.
        """
        legs: list[Position] = []
        for sym, pe in self.pending_entries.items():
            try:
                size = float(pe["size"])
                entry = float(pe["limit_price"])
                if size <= 0 or entry <= 0:
                    continue
                legs.append(
                    Position(
                        symbol=sym,
                        side=Side(pe["side"]),
                        entry_price=entry,
                        size=size,
                        leverage=int(pe.get("leverage") or 5),
                        stop_loss=float(pe.get("stop_loss") or 0),
                        take_profit=float(pe.get("take_profit") or 0),
                    )
                )
            except (KeyError, ValueError, TypeError) as e:
                # A malformed pending order must not silently vanish from the
                # budget — that is the bug this method exists to fix.
                self.logger.warning("PENDING_LEG_SKIPPED %s: %s", sym, e)
        return legs

    def _manage_pending_entries(self):
        if not self.pending_entries:
            return
        # Risk blocks must reach resting orders immediately, not on the model's
        # hourly cadence — a fill during a cooldown is exposure the risk engine
        # had already vetoed.
        blocked = self.risk.is_killed or (
            self.risk.cooldown_until and datetime.utcnow() < self.risk.cooldown_until
        )
        for symbol, pe in list(self.pending_entries.items()):
            try:
                if blocked:
                    self._abandon_pending(symbol, pe, "risk_blocked")
                else:
                    self._manage_one_pending(symbol, pe)
            except Exception as e:
                console.print(f"[red]Pending entry error {symbol}: {e}[/]")
                self.logger.exception("Pending %s: %s", symbol, e)

    def _manage_one_pending(self, symbol: str, pe: dict):
        side = Side(pe["side"])
        res = self.exchange.check_entry_fill(pe["order_id"], symbol)

        if res["status"] == "filled":
            self._on_entry_fill(
                symbol, pe,
                float(res.get("filled_amount") or pe["size"]),
                float(res.get("fill_price") or pe["limit_price"]),
            )
            return
        if res["status"] == "gone":
            filled = float(res.get("filled_amount") or 0)
            if filled > 0:
                self._on_entry_fill(
                    symbol, pe, filled,
                    float(res.get("fill_price") or pe["limit_price"]),
                )
            else:
                self.logger.info("ENTRY_GONE %s (order no longer exists)", symbol)
                self.pending_entries.pop(symbol, None)
                self._persist_state()
            return

        # Still resting: abandon if stale or run away from, else chase the touch.
        now = datetime.utcnow()
        try:
            placed_at = datetime.fromisoformat(str(pe["placed_at"]))
            last_place = datetime.fromisoformat(str(pe.get("last_reprice_at") or pe["placed_at"]))
        except ValueError:
            placed_at = last_place = now
        age = (now - placed_at).total_seconds()

        touch = self.exchange.touch_price(symbol, side)
        if touch <= 0:
            return

        atr = float(pe.get("atr") or 0)
        signal_price = float(pe["signal_price"])
        run = (touch - signal_price) if side == Side.LONG else (signal_price - touch)

        if age > self.entry_ttl_sec:
            self._abandon_pending(symbol, pe, "ttl_expired")
            return
        if atr > 0 and run > self.max_chase_atr * atr:
            self._abandon_pending(symbol, pe, "price_ran_away")
            return

        limit = float(pe["limit_price"])
        moved_away = (
            touch > limit * 1.0002 if side == Side.LONG else touch < limit * 0.9998
        )
        if not moved_away or (now - last_place).total_seconds() < self.reprice_sec:
            return

        # Chasing squeezes the reward side. If the R:R the model was approved on
        # no longer holds at the new price, the trade is over, not repriceable.
        sl, tp = float(pe["stop_loss"]), float(pe["take_profit"])
        stop_dist = abs(touch - sl)
        rr = abs(tp - touch) / stop_dist if stop_dist > 0 else 0
        if rr < float(pe.get("min_rr") or 0):
            self._abandon_pending(symbol, pe, f"rr_degraded_{rr:.2f}")
            return

        cancel = self.exchange.cancel_entry(pe["order_id"], symbol)
        filled = float(cancel.get("filled_amount") or 0)
        if filled > 0:
            self._on_entry_fill(symbol, pe, filled, float(pe["limit_price"]))
            return

        res = self.exchange.place_entry_limit(
            symbol, side, float(pe["size"]), touch,
            stop_loss=sl, take_profit=tp,
            strategy=pe.get("strategy") or "",
            leverage=int(pe.get("leverage") or 5),
            partial_take_profit=pe.get("partial_take_profit"),
            partial_fraction=float(pe.get("partial_fraction") or 0.5),
        )
        if "error" in res:
            self.logger.info("ENTRY_REPRICE_FAILED %s: %s", symbol, res["error"])
            self.pending_entries.pop(symbol, None)
            self._persist_state()
            return

        pe["order_id"] = str(res["id"])
        pe["limit_price"] = touch
        pe["last_reprice_at"] = now.isoformat()
        pe["reprices"] = int(pe.get("reprices") or 0) + 1
        self.logger.info(
            "ENTRY_REPRICE %s -> %.6f (#%d)", symbol, touch, pe["reprices"]
        )
        self._persist_state()

    def _on_entry_fill(self, symbol: str, pe: dict, filled_amount: float, fill_price: float):
        side = Side(pe["side"])
        fin = self.exchange.finalize_entry_fill(
            symbol, side, filled_amount, fill_price,
            stop_loss=float(pe["stop_loss"]),
            take_profit=float(pe["take_profit"]),
            strategy=pe.get("strategy") or "",
            partial_take_profit=pe.get("partial_take_profit"),
            partial_fraction=float(pe.get("partial_fraction") or 0.5),
        )
        console.print(
            f"[green]Maker entry filled: {pe['side']} {symbol} "
            f"{filled_amount:.6f} @ {fill_price:.6f} "
            f"(saved the spread + taker fee)[/]"
        )
        self.logger.info(
            "MAKER_FILL %s %s size=%.6f price=%.6f reprices=%d id=%s",
            pe["side"], symbol, filled_amount, fill_price,
            int(pe.get("reprices") or 0), pe["order_id"],
        )

        decision_id = pe.get("decision_id")
        if decision_id and self.decision_log:
            self.decision_log.link_order(
                decision_id,
                symbol=symbol,
                order_id=str(pe["order_id"]),
                side=pe["side"],
                size=filled_amount,
                entry_price=fill_price,
                stop_loss=float(pe["stop_loss"]),
                take_profit=float(pe["take_profit"]),
            )
            self.position_decisions[symbol] = decision_id
        self.position_conviction[symbol] = float(pe.get("strength") or 0.0)

        if fin.get("sl_placed") is False and self.exchange.mode != "paper":
            console.print("[yellow]   SL not on exchange — software stop active[/]")
        if fin.get("tp_placed") is False and self.exchange.mode != "paper":
            console.print("[yellow]   TP not on exchange — software TP active[/]")

        self.pending_entries.pop(symbol, None)
        self._persist_state()

    def _abandon_pending(self, symbol: str, pe: dict, reason: str):
        cancel = self.exchange.cancel_entry(pe["order_id"], symbol)
        filled = float(cancel.get("filled_amount") or 0)
        if filled > 0:
            # The cancel raced a fill. The money is in — bracket it, don't orphan it.
            self._on_entry_fill(symbol, pe, filled, float(pe["limit_price"]))
            return
        console.print(
            f"[yellow]Entry missed ({reason}): {symbol} never filled at "
            f"{float(pe['limit_price']):.6f} — no fee paid, no trade[/]"
        )
        self.logger.info(
            "ENTRY_MISSED %s reason=%s limit=%.6f reprices=%d",
            symbol, reason, float(pe["limit_price"]), int(pe.get("reprices") or 0),
        )
        self.pending_entries.pop(symbol, None)
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
                    # Only the TP leg can be maker: it is the one exit whose price
                    # we chose in advance and can rest at. Stops and trailing stops
                    # cross the book by construction and stay taker.
                    self._close_position(
                        position, current_price, "take_profit",
                        maker_price=self.exchange.maker_exit_price(position, current_price),
                    )
                elif position.should_trailing_stop(current_price):
                    console.print(f"[yellow]Trailing stop: {position.symbol}[/]")
                    self._close_position(position, current_price, "trailing_stop")

            except Exception as e:
                console.print(f"[red]Error managing {position.symbol}: {e}[/]")
                self.logger.exception("Manage %s: %s", position.symbol, e)

    def _close_position(
        self,
        position: Position,
        current_price: float,
        reason: str,
        *,
        maker_price: float | None = None,
    ):
        result = self.exchange.close_position(position.symbol, maker_price=maker_price)
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
            # Live: a proven maker exit rested at our own price, so it pays the
            # maker rate. Charging taker here would understate the live book
            # against the paper one and re-open the §2e comparability problem.
            rate = (
                self.exchange.maker_fee_rate
                if maker_price
                else self.exchange.commission_rate
            )
            exit_fee = position.size * exit_price * rate
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
        self.position_conviction.pop(position.symbol, None)
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
        # `exit=maker|taker` is what makes the maker-exit rule auditable after the
        # fact. §2e's lesson was that a fill rate nobody measured turned out to be
        # an artifact of the rule that granted it — so this one is recorded per
        # trade from the start, not reconstructed later.
        self.logger.info(
            "CLOSE %s %s pnl=%.2f banked=%.2f fees=%.4f reason=%s strategy=%s exit=%s",
            position.symbol, position.side.value, pnl,
            position.realized_pnl, fees, reason, position.strategy,
            "maker" if maker_price else "taker",
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
                    # Ditto for the conviction it was opened at, without which a
                    # restart would silently disarm the swap guard.
                    "position_conviction": self.position_conviction,
                    # Resting maker entries: restored on start, then reconciled
                    # against the venue by the first pending-management pass.
                    "pending_entries": self.pending_entries,
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
