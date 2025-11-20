import numpy as np
import pandas as pd
from indicators import *

def zscore(s: pd.Series, win: int=100) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std().replace(0, np.nan)
    return (s - m) / sd

def clip01(s: pd.Series) -> pd.Series:
    return s.clip(lower=0, upper=1)

def trend_features(df: pd.DataFrame, ref_df: pd.DataFrame=None):
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    ema10 = ema(close, 10)
    ema20 = ema(close, 20)
    slope200 = ema_slope(close, 200, lookback=5)
    adx14 = adx(high, low, close, 14)

    # Cross-asset: sector ETF confirmation trend (optional)
    if ref_df is not None:
        s_close = ref_df['Close']
        s_trend = (s_close > ema(s_close, 200)).astype(int)
    else:
        s_trend = pd.Series(index=close.index, data=np.nan)

    # Subscore: long-term above/below EMA, strength via slope/ADX, short-term alignment via EMA10/20
    long_term = (close > ema200).astype(int) * 0.6 + clip01(zscore(slope200, 200)).fillna(0)*0.2 + clip01(adx14/50).fillna(0)*0.2
    short_term = ((ema10 > ema20).astype(int) + (ema50 > ema200).astype(int))/2
    cross_asset = s_trend.fillna(0)  # 1 if sector is in LT uptrend

    trend_score = clip01(0.7*long_term + 0.2*short_term + 0.1*cross_asset)
    return trend_score

def momentum_features(df: pd.DataFrame):
    close, high, low = df['Close'], df['High'], df['Low']
    rsi14 = rsi(close, 14)
    stochr = stoch_rsi(close, 14)
    roc10 = roc(close, 10)
    bull_div = rsi_bullish_divergence(close, rsi14)
    bear_div = rsi_bearish_divergence(close, rsi14)

    # Dynamic thresholds: in strong trends (RSI>55), treat 60 as “momentum on”
    trend_bias = (rsi14 > 55).astype(int)
    rsi_component = np.where(trend_bias, (rsi14-40)/40, (rsi14-30)/40)  # maps into ~0..1
    rsi_component = pd.Series(rsi_component, index=close.index).clip(0,1)

    stoch_component = (stochr/100).clip(0,1)
    roc_component = clip01((roc10/5.0 + 0.5))  # -5%..+5% -> ~0..1

    # Divergence nudges
    div_nudge = (bull_div*0.05 - bear_div*0.05)

    mom_score = clip01(0.5*rsi_component + 0.3*stoch_component + 0.2*roc_component + div_nudge)
    return mom_score

def volume_features(df: pd.DataFrame):
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    obv_e = obv_ema(close, volume, span=10)
    vpt_s = vpt(close, volume)
    cmf20 = cmf(high, low, close, volume, 20)

    # Normalize volume to 20D average to find spikes
    vol_norm = volume / volume.rolling(20).mean()
    vol_spike = clip01((vol_norm - 1.0)/2.0)  # >1 means above avg; >3 is big

    # Combine conviction + flow + spike
    vscore = clip01(0.5*clip01(zscore(obv_e, 100)) + 0.3*clip01(zscore(vpt_s, 100)) + 0.2*clip01(cmf20*1.5) + 0.1*vol_spike)
    return vscore

def volatility_features(df: pd.DataFrame):
    close, high, low = df['Close'], df['High'], df['Low']
    atrp = atr_pct(high, low, close, 14)
    width = bb_width(close, 20, 2.0)
    expansion = volatility_expansion(atrp, base_window=50)

    # Prefer moderate ATR% (too high = choppy risk), but reward “expansion from low base”
    # Map ATR% roughly: 0.5%..3% -> good (0..1), >5% penalize
    atr_component = clip01((0.03 - atrp).clip(lower=-0.02, upper=0.03) / 0.03)  # lower ATR% better pre-breakout
    width_component = clip01((0.06 - width).clip(lower=-0.04, upper=0.06) / 0.06)  # compression
    expansion_nudge = expansion * 0.1

    vol_score = clip01(0.6*width_component + 0.3*atr_component + expansion_nudge)
    return vol_score

def breadth_features(market_df: pd.DataFrame, vix_df: pd.DataFrame=None, beta: float=1.0):
    m_close = market_df['Close']
    pct_above200 = percent_above_ema(m_close, 200).rolling(50).mean()  # proxy breadth: % of days above 200 (cheap)
    pct_above50  = percent_above_ema(m_close, 50).rolling(20).mean()

    breadth = clip01(0.6*pct_above200 + 0.4*pct_above50)

    if vix_df is not None:
        vix = vix_df['Close']
        vix_trend_down = (vix < ema(vix, 50)).astype(int)  # falling VIX = risk-on
        breadth = clip01(0.8*breadth + 0.2*vix_trend_down)

    # If low beta (<0.8), downweight breadth effect later (handled in signals)
    return breadth
