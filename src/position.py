# position.py
import pandas as pd
import numpy as np
from .indicators import atr_pct

def position_size(df: pd.DataFrame, risk_per_trade: float=0.01, max_leverage: float=1.0) -> pd.Series:
    close, high, low = df['Close'], df['High'], df['Low']
    atrp = atr_pct(high, low, close, 14).clip(lower=1e-6)
    # position size inversely proportional to volatility, capped
    raw = risk_per_trade / atrp
    sized = raw.clip(upper=max_leverage)
    return sized.fillna(0)

def stop_loss_levels(df: pd.DataFrame, multiple: float=2.0) -> pd.Series:
    close, high, low = df['Close'], df['High'], df['Low']
    atrp = atr_pct(high, low, close, 14)
    return close * (1 - multiple * atrp)  # trailing stop idea (you can convert to trailing separately)
