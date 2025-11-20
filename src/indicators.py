import numpy as np
import pandas as pd
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator, ROCIndicator, StochRSIIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def ema_slope(series: pd.Series, span: int, lookback: int=5) -> pd.Series:
    e = ema(series, span)
    # slope per-bar (simple difference); normalize by price to keep scale independent
    return e.diff(lookback) / (lookback * series)

def adx(high, low, close, window: int=14) -> pd.Series:
    return ADXIndicator(high, low, close, window=window).adx()

def rsi(close, window: int=14) -> pd.Series:
    return RSIIndicator(close, window=window).rsi()

def stoch_rsi(close, window: int=14, smooth1: int=3, smooth2: int=3) -> pd.Series:
    s = StochRSIIndicator(close, window=window, smooth1=smooth1, smooth2=smooth2)
    return s.stochrsi_k()  # 0..100

def roc(close, window: int=10) -> pd.Series:
    return ROCIndicator(close, window=window).roc()

def obv(close, volume) -> pd.Series:
    return OnBalanceVolumeIndicator(close, volume).on_balance_volume()

def obv_ema(close, volume, span: int=10) -> pd.Series:
    return obv(close, volume).ewm(span=span, adjust=False).mean()

def vpt(close, volume) -> pd.Series:
    # Volume-Price Trend: cumulative( volume * pct_change(close) )
    return (volume * close.pct_change().fillna(0)).cumsum()

def cmf(high, low, close, volume, window: int=20) -> pd.Series:
    return ChaikinMoneyFlowIndicator(high, low, close, volume, window=window).chaikin_money_flow()

def atr(high, low, close, window: int=14) -> pd.Series:
    return AverageTrueRange(high, low, close, window=window).average_true_range()

def atr_pct(high, low, close, window: int=14) -> pd.Series:
    a = atr(high, low, close, window)
    return a / close

def bb_width(close, window: int=20, n_std: float=2.0) -> pd.Series:
    bb = BollingerBands(close, window=window, window_dev=n_std)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    mid = bb.bollinger_mavg()
    return (upper - lower) / mid  # compression/expansion measure

def percent_above_ema(series: pd.Series, window: int=200) -> pd.Series:
    e = ema(series, window)
    return (series > e).astype(int)

def volatility_expansion(atrp: pd.Series, base_window: int=50) -> pd.Series:
    # Expansion when ATR% rises above its rolling median + small epsilon
    med = atrp.rolling(base_window).median()
    return (atrp > med * 1.05).astype(int)  # 5% above median -> “expanding”

def rsi_bullish_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int=60) -> pd.Series:
    """
    Very simple heuristic: price makes lower low while RSI makes higher low.
    Returns a boolean Series flagged near potential divergence points.
    """
    ll_price = close.rolling(lookback).apply(lambda x: np.argmin(x) == len(x)-1, raw=False).astype(bool)
    # Higher low in RSI: current RSI low higher than prior swing low (rough heuristic)
    rsi_min = rsi_series.rolling(lookback).min()
    higher_low = rsi_series > rsi_min.shift(lookback//2)  # crude but effective enough for filtering
    return (ll_price & higher_low).astype(int)

def rsi_bearish_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int=60) -> pd.Series:
    hh_price = close.rolling(lookback).apply(lambda x: np.argmax(x) == len(x)-1, raw=False).astype(bool)
    rsi_max = rsi_series.rolling(lookback).max()
    lower_high = rsi_series < rsi_max.shift(lookback//2)
    return (hh_price & lower_high).astype(int)
