import numpy as np
import ccxt
from datetime import datetime, timedelta
from src.indicators.technical import calculate_ema

exchange = ccxt.binance({'enableRateLimit': True})
since = int((datetime.utcnow() - timedelta(days=90)).timestamp() * 1000)
ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', since=since, limit=1000)
closes = np.array([c[4] for c in ohlcv])

ema_9 = calculate_ema(closes, 9)
ema_21 = calculate_ema(closes, 21)
ema_50 = calculate_ema(closes, 50)
ema_100 = calculate_ema(closes, 100)

print(f'Last 5 candles:')
for i in range(-5, 0):
    print(f'  EMA9={ema_9[i]:.1f}, EMA21={ema_21[i]:.1f}, EMA50={ema_50[i]:.1f}, EMA100={ema_100[i]:.1f}')
    print(f'  9>21: {ema_9[i] > ema_21[i]}, 21>50: {ema_21[i] > ema_50[i]}, 50>100: {ema_50[i] > ema_100[i]}')

aligned_bull = 0
aligned_bear = 0
for i in range(100, len(closes)):
    if ema_9[i] > ema_21[i] and ema_21[i] > ema_50[i] and ema_50[i] > ema_100[i]:
        aligned_bull += 1
    if ema_9[i] < ema_21[i] and ema_21[i] < ema_50[i] and ema_50[i] < ema_100[i]:
        aligned_bear += 1

total = len(closes) - 100
print(f'\nBullish aligned: {aligned_bull}/{total} ({aligned_bull/total*100:.1f}%)')
print(f'Bearish aligned: {aligned_bear}/{total} ({aligned_bear/total*100:.1f}%)')
