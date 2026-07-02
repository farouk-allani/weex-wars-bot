"""Debug: check funding rate distribution"""
import ccxt
import numpy as np
from datetime import datetime, timedelta

exchange = ccxt.binance({'enableRateLimit': True})
since = int((datetime.utcnow() - timedelta(days=90)).timestamp() * 1000)
data = exchange.fetch_funding_rate_history('BTC/USDT:USDT', since=since, limit=1000)
rates = [d.get('fundingRate', 0) for d in data]

print(f'Funding rates: {len(rates)} samples')
print(f'Mean: {np.mean(rates):.6f}')
print(f'Std: {np.std(rates):.6f}')
print(f'Min: {np.min(rates):.6f}')
print(f'Max: {np.max(rates):.6f}')
print(f'Values > 0.001: {sum(1 for r in rates if r > 0.001)}')
print(f'Values < -0.001: {sum(1 for r in rates if r < -0.001)}')
print(f'Values > 0.0005: {sum(1 for r in rates if r > 0.0005)}')
print(f'Values < -0.0005: {sum(1 for r in rates if r < -0.0005)}')
print(f'Sample values: {rates[:10]}')
