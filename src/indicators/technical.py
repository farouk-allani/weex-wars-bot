"""WEEX AI Wars II — Technical Indicators"""

import numpy as np
import pandas as pd
from typing import Optional

from ..core.models import Candle, MarketRegime


def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)

    # Initial averages
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Smoothed averages (Wilder's method)
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50  # Fill initial values
    return rsi


def calculate_macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD, Signal, Histogram."""
    ema_fast = calculate_ema(closes, fast)
    ema_slow = calculate_ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    ema = np.zeros_like(data, dtype=float)
    multiplier = 2 / (period + 1)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

    return ema


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    sma = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1 : i + 1])
    return sma


def calculate_bollinger_bands(
    closes: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper band, Middle band, Lower band."""
    middle = calculate_sma(closes, period)

    std = np.full_like(closes, np.nan, dtype=float)
    for i in range(period - 1, len(closes)):
        std[i] = np.std(closes[i - period + 1 : i + 1])

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return upper, middle, lower


def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range."""
    tr = np.zeros_like(closes)
    tr[0] = highs[0] - lows[0]

    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    atr = np.zeros_like(closes)
    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def calculate_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index."""
    n = len(closes)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Smoothed values
    atr = calculate_atr(highs, lows, closes, period)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    plus_dm_smooth[period] = np.sum(plus_dm[1 : period + 1])
    minus_dm_smooth[period] = np.sum(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - plus_dm_smooth[i - 1] / period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - minus_dm_smooth[i - 1] / period + minus_dm[i]

    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i] / period
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i] / period

    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    adx = np.zeros(n)
    adx[2 * period - 1] = np.mean(dx[period : 2 * period])
    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def calculate_vwap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Volume Weighted Average Price (rolling)."""
    typical_price = (highs + lows + closes) / 3
    vwap = np.full_like(closes, np.nan, dtype=float)

    for i in range(period - 1, len(closes)):
        start = i - period + 1
        tp_vol = np.sum(typical_price[start : i + 1] * volumes[start : i + 1])
        vol_sum = np.sum(volumes[start : i + 1])
        if vol_sum > 0:
            vwap[i] = tp_vol / vol_sum

    return vwap


def calculate_stochastic_rsi(
    closes: np.ndarray, rsi_period: int = 14, stoch_period: int = 14,
    k_period: int = 3, d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI — %K and %D."""
    rsi = calculate_rsi(closes, rsi_period)

    stoch_rsi = np.zeros_like(rsi)
    for i in range(stoch_period, len(rsi)):
        rsi_window = rsi[i - stoch_period + 1 : i + 1]
        rsi_min = np.min(rsi_window)
        rsi_max = np.max(rsi_window)
        if rsi_max - rsi_min > 0:
            stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) * 100
        else:
            stoch_rsi[i] = 50

    k = calculate_sma(stoch_rsi, k_period)
    d = calculate_sma(k, d_period)

    return k, d


# ---- Market Regime Detection ----

def detect_regime(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    adx_period: int = 14,
    adx_threshold: float = 25,
) -> MarketRegime:
    """Detect current market regime using ADX and volatility."""
    adx = calculate_adx(highs, lows, closes, adx_period)
    current_adx = adx[-1] if len(adx) > 0 else 0

    # Check recent volatility
    atr = calculate_atr(highs, lows, closes, 14)
    recent_atr = atr[-1] if len(atr) > 0 else 0
    avg_atr = np.mean(atr[-50:]) if len(atr) >= 50 else recent_atr

    # Check trend direction with EMA
    ema_fast = calculate_ema(closes, 9)
    ema_slow = calculate_ema(closes, 21)

    if current_adx >= adx_threshold:
        if ema_fast[-1] > ema_slow[-1]:
            return MarketRegime.TRENDING_UP
        else:
            return MarketRegime.TRENDING_DOWN
    elif recent_atr > avg_atr * 1.5:
        return MarketRegime.HIGH_VOLATILITY
    else:
        return MarketRegime.RANGING
