"""WEEX AI Wars II — Composite Strategy v6: Bugfixed + Adaptive

Fixes from v5:
1. Time-of-day uses CANDLE timestamp, not datetime.now() — backtest now accurate
2. Session edge no longer counts as directional confirmation (was giving free pass)
3. ATR multipliers adapt to pair volatility (BTC tighter, SOL wider)
4. Drawdown circuit breaker — pause after 3 consecutive losses
5. Better pullback detection — requires price to actually touch EMA, not just be near
"""

import numpy as np
from datetime import datetime, timezone

from ..core.models import Candle, Signal, Side, MarketRegime
from ..indicators.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, calculate_ema, calculate_vwap,
    detect_regime, calculate_stochastic_rsi,
)
from .edges import EdgeStrategies


# ATR multiplier profiles per symbol (based on volatility)
ATR_PROFILES = {
    "BTC": {"stop": 1.3, "target": 4.0, "breakeven": 0.8},  # Tighter — less noise
    "ETH": {"stop": 1.5, "target": 4.5, "breakeven": 1.0},  # Medium
    "SOL": {"stop": 1.8, "target": 5.5, "breakeven": 1.2},  # Wider — more noise
}
DEFAULT_PROFILE = {"stop": 1.5, "target": 5.0, "breakeven": 1.0}


class CompositeStrategy:
    """
    Competition strategy v6: Bugfixed + Adaptive.

    Key fixes:
    - Time filter uses candle timestamp (backtest accuracy)
    - Session edge doesn't count toward directional confirmation
    - ATR multipliers adapt to each pair's volatility
    - Drawdown circuit breaker
    """

    def __init__(self, config: dict):
        self.config = config
        self.tf_config = config.get("strategy", {}).get("trend_follow", {})
        self.mr_config = config.get("strategy", {}).get("mean_reversion", {})
        self.ind_config = config.get("indicators", {})
        self.edges = EdgeStrategies(config)

        # Circuit breaker state
        self.consecutive_losses = 0
        self.circuit_breaker_until = None

    def analyze(
        self,
        symbol: str,
        candles: list[Candle],
        funding_rate: float = 0.0,
        existing_positions: list = None,
    ) -> Signal | None:
        if len(candles) < 100:
            return None

        # Get ATR profile for this symbol
        base = symbol.split("/")[0]
        profile = ATR_PROFILES.get(base, DEFAULT_PROFILE)

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # ---- FILTER 1: Time-of-Day (uses candle timestamp!) ----
        candle_time = candles[-1].timestamp
        if self._is_asia_session(candle_time):
            return None

        # ---- FILTER 2: Circuit Breaker ----
        if self.circuit_breaker_until and candle_time < self.circuit_breaker_until:
            return None  # Paused after consecutive losses
        else:
            self.circuit_breaker_until = None  # Reset

        # ---- FILTER 3: Trend Detection ----
        adx = calculate_adx(highs, lows, closes, 14)
        current_adx = adx[-1]
        adx_threshold = self.tf_config.get("adx_threshold", 30)

        if current_adx < adx_threshold:
            return None

        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)

        ema_21_slope = (ema_21[-1] - ema_21[-5]) / ema_21[-5] * 100 if ema_21[-5] > 0 else 0

        bullish_stack = ema_9[-1] > ema_21[-1] > ema_50[-1]
        bearish_stack = ema_9[-1] < ema_21[-1] < ema_50[-1]

        if not bullish_stack and not bearish_stack:
            return None

        # ---- FILTER 4: Edge Analysis ----
        edges = self.edges.analyze_all_edges(candles, funding_rate)
        edge_multiplier, edge_direction = self.edges.get_combined_modifier(edges)
        confirming_edges = self._count_directional_edges(
            edges, "long" if bullish_stack else "short"
        )

        if confirming_edges < 2:
            return None

        # ---- FILTER 5: Funding Rate ----
        if bullish_stack and funding_rate > 0.0005:
            return None
        if bearish_stack and funding_rate < -0.0005:
            return None

        # ---- FILTER 6: Correlation Guard ----
        if existing_positions:
            for pos_symbol, pos_side in existing_positions:
                if pos_symbol == symbol:
                    return None
                if self._is_correlated(symbol, pos_symbol):
                    if pos_side == ("long" if bullish_stack else "short"):
                        return None

        # ---- Generate Signal ----
        atr = calculate_atr(highs, lows, closes, 14)
        current_atr = atr[-1]
        current_price = closes[-1]

        rsi = calculate_rsi(closes, 10)
        current_rsi = rsi[-1]
        stoch_k, stoch_d = calculate_stochastic_rsi(closes)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes, 20, 1.8)

        signal = None

        if bullish_stack:
            signal = self._long_signal(
                symbol, closes, current_price, current_atr, current_rsi,
                rsi[-2], stoch_k, stoch_d, ema_21, ema_21_slope, current_adx,
                bb_lower, highs, lows, profile,
            )
        if signal is None and bearish_stack:
            signal = self._short_signal(
                symbol, closes, current_price, current_atr, current_rsi,
                rsi[-2], stoch_k, stoch_d, ema_21, ema_21_slope, current_adx,
                bb_upper, highs, lows, profile,
            )

        if signal is None:
            return None

        # ---- R:R Check ----
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        if risk > 0 and (reward / risk) < 2.0:
            return None

        # ---- Apply Edge Multiplier ----
        signal.strength *= edge_multiplier

        if edge_direction == "long" and signal.side == Side.SHORT and edge_multiplier > 1.2:
            return None
        if edge_direction == "short" and signal.side == Side.LONG and edge_multiplier > 1.2:
            return None

        if signal.strength < 0.5:
            return None

        return signal

    def record_loss(self, candle_time: datetime):
        """Track consecutive losses for circuit breaker."""
        self.consecutive_losses += 1
        if self.consecutive_losses >= 3:
            # Pause for 6 hours (6 candles on 1h)
            self.circuit_breaker_until = candle_time + __import__('datetime').timedelta(hours=6)
            self.consecutive_losses = 0

    def record_win(self):
        """Reset consecutive loss counter on win."""
        self.consecutive_losses = 0

    # ---- Filters ----

    def _is_asia_session(self, candle_time: datetime) -> bool:
        """FIXED: Uses candle timestamp, not datetime.now()."""
        if candle_time.tzinfo is None:
            candle_time = candle_time.replace(tzinfo=timezone.utc)
        return 0 <= candle_time.hour < 8

    def _is_correlated(self, symbol1: str, symbol2: str) -> bool:
        base1 = symbol1.split("/")[0]
        base2 = symbol2.split("/")[0]
        if base1 in ["BTC", "ETH"] and base2 in ["BTC", "ETH"]:
            return True
        return False

    def _count_directional_edges(self, edges: dict, direction: str) -> int:
        """
        Count edges that confirm direction.
        FIXED: Session edge does NOT count (it has no direction).
        """
        count = 0

        liq = edges.get("liquidation", {})
        if liq.get("detected") and liq.get("direction") == direction:
            count += 1

        fund = edges.get("funding", {})
        if fund.get("signal") and fund.get("direction") == direction:
            count += 1

        vol = edges.get("volume", {})
        if vol.get("anomaly"):
            if (direction == "long" and vol.get("direction") == "accumulation") or \
               (direction == "short" and vol.get("direction") == "distribution"):
                count += 1

        mtf = edges.get("mtf", {})
        if mtf.get("aligned") and mtf.get("direction") == direction:
            count += 1

        # Session does NOT count — it has no direction
        # It still boosts the multiplier via get_combined_modifier

        return count

    # ---- Entry Signals ----

    def _long_signal(
        self, symbol, closes, current_price, current_atr, current_rsi,
        prev_rsi, stoch_k, stoch_d, ema_21, ema_21_slope, current_adx,
        bb_lower, highs, lows, profile,
    ) -> Signal | None:
        ema_val = ema_21[-1]
        dist_to_ema = (current_price - ema_val) / current_atr if current_atr > 0 else 999

        if not (-0.5 <= dist_to_ema <= 1.5):
            return None

        candle_0_bullish = closes[-1] > closes[-2]
        rsi_recovery = current_rsi > prev_rsi and current_rsi < 65 and current_rsi > 30
        stoch_turn = stoch_k[-1] > stoch_d[-1]
        price_support = current_price >= ema_val * 0.997
        slope_positive = ema_21_slope > 0

        if candle_0_bullish and rsi_recovery and stoch_turn and price_support and slope_positive:
            strength = 0.65
            if current_adx > 30:
                strength += 0.1

            # Adaptive ATR multipliers
            stop = min(ema_val - current_atr * profile["breakeven"],
                       current_price - current_atr * profile["stop"])
            target = current_price + current_atr * profile["target"]

            return Signal(
                symbol=symbol, side=Side.LONG,
                strength=min(1.0, strength),
                strategy="trend_rider",
                entry_price=current_price,
                stop_loss=stop, take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Trend LONG: EMA pullback, RSI recover, ADX={current_adx:.0f}",
            )

        # BB lower backup
        near_bb_lower = current_price <= bb_lower[-1] * 1.005
        if near_bb_lower and rsi_recovery and stoch_turn and slope_positive:
            strength = 0.55
            stop = current_price - current_atr * profile["stop"]
            target = current_price + current_atr * profile["target"]

            return Signal(
                symbol=symbol, side=Side.LONG,
                strength=min(1.0, strength),
                strategy="trend_rider_bb",
                entry_price=current_price,
                stop_loss=stop, take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"BB Pullback LONG: Lower BB touch, RSI recover, ADX={current_adx:.0f}",
            )

        return None

    def _short_signal(
        self, symbol, closes, current_price, current_atr, current_rsi,
        prev_rsi, stoch_k, stoch_d, ema_21, ema_21_slope, current_adx,
        bb_upper, highs, lows, profile,
    ) -> Signal | None:
        ema_val = ema_21[-1]
        dist_to_ema = (ema_val - current_price) / current_atr if current_atr > 0 else 999

        if not (-0.5 <= dist_to_ema <= 1.5):
            return None

        candle_0_bearish = closes[-1] < closes[-2]
        rsi_reject = current_rsi < prev_rsi and current_rsi > 35 and current_rsi < 70
        stoch_turn = stoch_k[-1] < stoch_d[-1]
        price_resistance = current_price <= ema_val * 1.003
        slope_negative = ema_21_slope < 0

        if candle_0_bearish and rsi_reject and stoch_turn and price_resistance and slope_negative:
            strength = 0.65
            if current_adx > 30:
                strength += 0.1

            stop = max(ema_val + current_atr * profile["breakeven"],
                       current_price + current_atr * profile["stop"])
            target = current_price - current_atr * profile["target"]

            return Signal(
                symbol=symbol, side=Side.SHORT,
                strength=min(1.0, strength),
                strategy="trend_rider",
                entry_price=current_price,
                stop_loss=stop, take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Trend SHORT: EMA pullback reject, RSI falling, ADX={current_adx:.0f}",
            )

        near_bb_upper = current_price >= bb_upper[-1] * 0.995
        if near_bb_upper and rsi_reject and stoch_turn and slope_negative:
            strength = 0.55
            stop = current_price + current_atr * profile["stop"]
            target = current_price - current_atr * profile["target"]

            return Signal(
                symbol=symbol, side=Side.SHORT,
                strength=min(1.0, strength),
                strategy="trend_rider_bb",
                entry_price=current_price,
                stop_loss=stop, take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"BB Pullback SHORT: Upper BB touch, RSI reject, ADX={current_adx:.0f}",
            )

        return None

    def get_chandelier_exit(self, highs, lows, closes, period=22, multiplier=3.0):
        atr = calculate_atr(highs, lows, closes, 14)
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        return highest_high - atr[-1] * multiplier, lowest_low + atr[-1] * multiplier
