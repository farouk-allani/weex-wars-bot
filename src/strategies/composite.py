"""WEEX AI Wars II — Composite Strategy v4: Trend Rider

Philosophy: Don't predict where price goes. Ride it there.

The competition rewards RETURNS, not win rate. A 30% win rate with 5:1 R:R
beats a 60% win rate with 1:1 R:R.

Strategy: Enter strong trends, ride them with a trailing stop.
- Enter on pullback in strong trend (ADX > 25, EMA stack)
- NO fixed take-profit — let the trend run
- Trailing stop at 2x ATR — wide enough to survive noise, tight enough to lock profits
- Minimum 2 edges confirming
- R:R must be >= 2:1 based on initial stop only
"""

import numpy as np
from datetime import datetime

from ..core.models import Candle, Signal, Side, MarketRegime
from ..indicators.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, calculate_ema, calculate_vwap,
    detect_regime, calculate_stochastic_rsi,
)
from .edges import EdgeStrategies


class CompositeStrategy:
    """
    Trend Rider: Enter pullbacks in strong trends, ride with trailing stop.
    No fixed take-profit — let winners run.
    """

    def __init__(self, config: dict):
        self.config = config
        self.tf_config = config.get("strategy", {}).get("trend_follow", {})
        self.mr_config = config.get("strategy", {}).get("mean_reversion", {})
        self.ind_config = config.get("indicators", {})
        self.edges = EdgeStrategies(config)

    def analyze(self, symbol: str, candles: list[Candle], funding_rate: float = 0.0) -> Signal | None:
        if len(candles) < 100:
            return None

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # ---- Trend Detection ----
        adx = calculate_adx(highs, lows, closes, 14)
        current_adx = adx[-1]
        adx_threshold = self.tf_config.get("adx_threshold", 25)

        if current_adx < adx_threshold:
            return None

        # EMA stack
        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        ema_50 = calculate_ema(closes, 50)
        ema_100 = calculate_ema(closes, 100)

        # Check EMA slope (trend must be moving, not flat)
        ema_21_slope = (ema_21[-1] - ema_21[-5]) / ema_21[-5] * 100 if ema_21[-5] > 0 else 0

        bullish_stack = ema_9[-1] > ema_21[-1] > ema_50[-1]
        bearish_stack = ema_9[-1] < ema_21[-1] < ema_50[-1]

        if not bullish_stack and not bearish_stack:
            return None

        # ---- Edge Analysis ----
        edges = self.edges.analyze_all_edges(candles, funding_rate)
        edge_multiplier, edge_direction = self.edges.get_combined_modifier(edges)
        confirming_edges = self._count_confirming_edges(
            edges, "long" if bullish_stack else "short"
        )

        if confirming_edges < 2:
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

        # ---- LONG ----
        if bullish_stack:
            signal = self._long_signal(
                symbol, closes, current_price, current_atr, current_rsi,
                rsi[-2], stoch_k, stoch_d, ema_21, ema_21_slope, current_adx, bb_lower
            )

        # ---- SHORT ----
        if signal is None and bearish_stack:
            signal = self._short_signal(
                symbol, closes, current_price, current_atr, current_rsi,
                rsi[-2], stoch_k, stoch_d, ema_21, ema_21_slope, current_adx, bb_upper
            )

        if signal is None:
            return None

        # ---- R:R Check (initial stop only — no fixed TP) ----
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

    def _long_signal(
        self, symbol, closes, current_price, current_atr, current_rsi,
        prev_rsi, stoch_k, stoch_d, ema_21, ema_21_slope, current_adx, bb_lower,
    ) -> Signal | None:
        """
        Long entry: Pullback to 21 EMA or BB lower in uptrend.
        - Price near support (EMA or BB)
        - RSI recovering from dip
        - Trend slope positive
        """
        ema_val = ema_21[-1]
        dist_to_ema = (current_price - ema_val) / current_atr if current_atr > 0 else 999

        # Condition 1: Price near 21 EMA (within 1.5 ATR above, or touching from above)
        near_ema = -0.5 <= dist_to_ema <= 1.5

        # Condition 2: RSI was dipping and now recovering
        rsi_dip_recovery = current_rsi > prev_rsi and current_rsi < 65 and current_rsi > 30

        # Condition 3: StochRSI turning up
        stoch_turn = stoch_k[-1] > stoch_d[-1]

        # Condition 4: Price above or bouncing off support
        price_support = current_price >= ema_val * 0.997

        # Condition 5: Trend slope must be positive (trend is moving, not flat)
        slope_positive = ema_21_slope > 0

        if near_ema and rsi_dip_recovery and stoch_turn and price_support and slope_positive:
            strength = 0.6

            # Stronger ADX = more conviction
            if current_adx > 30:
                strength += 0.1

            # Stop below EMA with buffer
            stop = min(ema_val - current_atr * 0.8, current_price - current_atr * 1.5)

            # NO fixed take-profit — set it very wide, trailing stop will manage exit
            # Set TP at 5x ATR (trailing should trigger before this)
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
                reason=f"Trend Rider LONG: Pullback to EMA({ema_val:.0f}), RSI recover, ADX={current_adx:.0f}, slope={ema_21_slope:.2f}%",
            )

        # Also try BB lower touch
        near_bb_lower = current_price <= bb_lower[-1] * 1.005
        if near_bb_lower and rsi_dip_recovery and stoch_turn and slope_positive:
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
                reason=f"Trend Rider LONG: BB lower touch, RSI recover, ADX={current_adx:.0f}",
            )

        return None

    def _short_signal(
        self, symbol, closes, current_price, current_atr, current_rsi,
        prev_rsi, stoch_k, stoch_d, ema_21, ema_21_slope, current_adx, bb_upper,
    ) -> Signal | None:
        """
        Short entry: Pullback to 21 EMA or BB upper in downtrend.
        """
        ema_val = ema_21[-1]
        dist_to_ema = (ema_val - current_price) / current_atr if current_atr > 0 else 999

        near_ema = -0.5 <= dist_to_ema <= 1.5
        rsi_rally_reject = current_rsi < prev_rsi and current_rsi > 35 and current_rsi < 70
        stoch_turn = stoch_k[-1] < stoch_d[-1]
        price_resistance = current_price <= ema_val * 1.003
        slope_negative = ema_21_slope < 0

        if near_ema and rsi_rally_reject and stoch_turn and price_resistance and slope_negative:
            strength = 0.6

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
                reason=f"Trend Rider SHORT: Pullback to EMA({ema_val:.0f}), RSI reject, ADX={current_adx:.0f}, slope={ema_21_slope:.2f}%",
            )

        near_bb_upper = current_price >= bb_upper[-1] * 0.995
        if near_bb_upper and rsi_rally_reject and stoch_turn and slope_negative:
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
                reason=f"Trend Rider SHORT: BB upper touch, RSI reject, ADX={current_adx:.0f}",
            )

        return None
