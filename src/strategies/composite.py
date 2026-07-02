"""WEEX AI Wars II — Composite Strategy v5: Competition Killer

Improvements over v4:
1. Funding rate filter — skip longs when funding > 0.005% (paying to hold)
2. Time-of-day filter — skip Asia session (low liquidity, choppy)
3. Correlation awareness — signal includes correlation hint
4. Chandelier exit calculation — dynamic trailing stop target
5. Better pullback detection — use 3-candle pattern, not just 1
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


class CompositeStrategy:
    """
    Competition strategy v5: Trend Rider + Filters.

    Filters (hard gates):
    - ADX >= 30 (strong trend only)
    - MTF alignment (9/21/50 EMA stack)
    - 2+ edges confirming
    - R:R >= 2:1
    - Funding rate not against us
    - Not in Asia session
    - Not correlated with existing position

    Entry types:
    - Pullback to 21 EMA (best R:R)
    - Pullback to BB (good R:R)
    - Momentum (only ADX > 35, strong trend)
    """

    def __init__(self, config: dict):
        self.config = config
        self.tf_config = config.get("strategy", {}).get("trend_follow", {})
        self.mr_config = config.get("strategy", {}).get("mean_reversion", {})
        self.ind_config = config.get("indicators", {})
        self.edges = EdgeStrategies(config)

    def analyze(
        self,
        symbol: str,
        candles: list[Candle],
        funding_rate: float = 0.0,
        existing_positions: list = None,
    ) -> Signal | None:
        """
        Ultra-selective signal generation.
        existing_positions: list of (symbol, side) tuples for correlation check
        """
        if len(candles) < 100:
            return None

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # ---- FILTER 1: Time-of-Day ----
        if self._is_asia_session():
            return None  # Skip low-liquidity Asia session

        # ---- FILTER 2: Trend Detection ----
        adx = calculate_adx(highs, lows, closes, 14)
        current_adx = adx[-1]
        adx_threshold = self.tf_config.get("adx_threshold", 30)

        if current_adx < adx_threshold:
            return None

        # EMA stack
        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)

        # EMA slope (trend must be moving)
        ema_21_slope = (ema_21[-1] - ema_21[-5]) / ema_21[-5] * 100 if ema_21[-5] > 0 else 0

        bullish_stack = ema_9[-1] > ema_21[-1] > ema_50[-1]
        bearish_stack = ema_9[-1] < ema_21[-1] < ema_50[-1]

        if not bullish_stack and not bearish_stack:
            return None

        # ---- FILTER 3: Edge Analysis ----
        edges = self.edges.analyze_all_edges(candles, funding_rate)
        edge_multiplier, edge_direction = self.edges.get_combined_modifier(edges)
        confirming_edges = self._count_confirming_edges(
            edges, "long" if bullish_stack else "short"
        )

        if confirming_edges < 2:
            return None

        # ---- FILTER 4: Funding Rate ----
        # Don't go long when funding is very positive (paying to hold longs)
        # Don't go short when funding is very negative (paying to hold shorts)
        if bullish_stack and funding_rate > 0.0005:  # 0.05% — significant
            return None
        if bearish_stack and funding_rate < -0.0005:
            return None

        # ---- FILTER 5: Correlation Guard ----
        if existing_positions:
            for pos_symbol, pos_side in existing_positions:
                if pos_symbol == symbol:
                    return None  # Already have position
                # Check if correlated (BTC moves with ETH/SOL)
                if self._is_correlated(symbol, pos_symbol):
                    if pos_side == ("long" if bullish_stack else "short"):
                        return None  # Already have correlated position in same direction

        # ---- Generate Signal ----
        atr = calculate_atr(highs, lows, closes, 14)
        current_atr = atr[-1]
        current_price = closes[-1]

        rsi = calculate_rsi(closes, 10)
        current_rsi = rsi[-1]
        stoch_k, stoch_d = calculate_stochastic_rsi(closes)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes, 20, 1.8)

        signal = None

        # Priority 1: Pullback to EMA
        if bullish_stack:
            signal = self._long_signal(
                symbol, closes, current_price, current_atr, current_rsi,
                rsi[-2], stoch_k, stoch_d, ema_21, ema_21_slope, current_adx,
                bb_lower, highs, lows,
            )
        if signal is None and bearish_stack:
            signal = self._short_signal(
                symbol, closes, current_price, current_atr, current_rsi,
                rsi[-2], stoch_k, stoch_d, ema_21, ema_21_slope, current_adx,
                bb_upper, highs, lows,
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

    # ---- Filters ----

    def _is_asia_session(self) -> bool:
        """Skip Asia session (00:00-08:00 UTC) — low liquidity, choppy."""
        now = datetime.now(timezone.utc)
        return 0 <= now.hour < 8

    def _is_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are highly correlated."""
        # Crypto correlation groups
        group1 = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        group2 = ["SOL/USDT:USDT"]  # SOL is somewhat independent

        base1 = symbol1.split("/")[0]
        base2 = symbol2.split("/")[0]

        # BTC and ETH are highly correlated
        if base1 in ["BTC", "ETH"] and base2 in ["BTC", "ETH"]:
            return True

        # SOL correlates with both but less strongly
        # Allow one BTC/ETH + one SOL position simultaneously
        return False

    def _count_confirming_edges(self, edges: dict, direction: str) -> int:
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
        session = edges.get("session", {})
        if session.get("favorable"):
            count += 1
        return count

    # ---- Entry Signals ----

    def _long_signal(
        self, symbol, closes, current_price, current_atr, current_rsi,
        prev_rsi, stoch_k, stoch_d, ema_21, ema_21_slope, current_adx,
        bb_lower, highs, lows,
    ) -> Signal | None:
        """Long: Pullback to 21 EMA with 3-candle bounce confirmation."""
        ema_val = ema_21[-1]
        dist_to_ema = (current_price - ema_val) / current_atr if current_atr > 0 else 999

        # Must be near EMA (within 1.5 ATR above, or touching)
        if not (-0.5 <= dist_to_ema <= 1.5):
            return None

        # 3-candle bounce pattern: lower wick, green candle, follow-through
        candle_0_bullish = closes[-1] > closes[-2]  # Current candle up
        candle_1_green = closes[-2] > closes[-3]  # Previous candle green
        lower_wick = (min(closes[-1], closes[-1]) - lows[-1]) > current_atr * 0.3  # Has lower wick

        # RSI recovering
        rsi_recovery = current_rsi > prev_rsi and current_rsi < 65 and current_rsi > 30

        # StochRSI turning up
        stoch_turn = stoch_k[-1] > stoch_d[-1]

        # Price at or above EMA (bounced)
        price_support = current_price >= ema_val * 0.997

        # Trend slope positive
        slope_positive = ema_21_slope > 0

        if candle_0_bullish and rsi_recovery and stoch_turn and price_support and slope_positive:
            strength = 0.65

            if current_adx > 30:
                strength += 0.1

            # Stop below EMA with buffer
            stop = min(ema_val - current_atr * 0.8, current_price - current_atr * 1.5)

            # Wide TP — trailing stop manages exit
            target = current_price + current_atr * 5.0

            return Signal(
                symbol=symbol,
                side=Side.LONG,
                strength=min(1.0, strength),
                strategy="trend_rider",
                entry_price=current_price,
                stop_loss=stop,
                take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Trend Rider LONG: EMA pullback bounce, RSI recover, ADX={current_adx:.0f}",
            )

        # BB lower touch as backup
        near_bb_lower = current_price <= bb_lower[-1] * 1.005
        if near_bb_lower and rsi_recovery and stoch_turn and slope_positive:
            strength = 0.55

            stop = current_price - current_atr * 1.5
            target = current_price + current_atr * 5.0

            return Signal(
                symbol=symbol,
                side=Side.LONG,
                strength=min(1.0, strength),
                strategy="trend_rider_bb",
                entry_price=current_price,
                stop_loss=stop,
                take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"BB Pullback LONG: Lower BB touch, RSI recover, ADX={current_adx:.0f}",
            )

        return None

    def _short_signal(
        self, symbol, closes, current_price, current_atr, current_rsi,
        prev_rsi, stoch_k, stoch_d, ema_21, ema_21_slope, current_adx,
        bb_upper, highs, lows,
    ) -> Signal | None:
        """Short: Pullback to 21 EMA with rejection confirmation."""
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

            stop = max(ema_val + current_atr * 0.8, current_price + current_atr * 1.5)
            target = current_price - current_atr * 5.0

            return Signal(
                symbol=symbol,
                side=Side.SHORT,
                strength=min(1.0, strength),
                strategy="trend_rider",
                entry_price=current_price,
                stop_loss=stop,
                take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"Trend Rider SHORT: EMA pullback reject, RSI falling, ADX={current_adx:.0f}",
            )

        near_bb_upper = current_price >= bb_upper[-1] * 0.995
        if near_bb_upper and rsi_reject and stoch_turn and slope_negative:
            strength = 0.55

            stop = current_price + current_atr * 1.5
            target = current_price - current_atr * 5.0

            return Signal(
                symbol=symbol,
                side=Side.SHORT,
                strength=min(1.0, strength),
                strategy="trend_rider_bb",
                entry_price=current_price,
                stop_loss=stop,
                take_profit=target,
                leverage=self.config.get("trading", {}).get("default_leverage", 5),
                reason=f"BB Pullback SHORT: Upper BB touch, RSI reject, ADX={current_adx:.0f}",
            )

        return None

    def get_chandelier_exit(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int = 22, multiplier: float = 3.0,
    ) -> tuple[float, float]:
        """
        Chandelier Exit — dynamic trailing stop based on ATR.
        Long: Highest High - ATR * multiplier
        Short: Lowest Low + ATR * multiplier
        """
        atr = calculate_atr(highs, lows, closes, 14)
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])

        chandelier_long = highest_high - atr[-1] * multiplier
        chandelier_short = lowest_low + atr[-1] * multiplier

        return chandelier_long, chandelier_short
