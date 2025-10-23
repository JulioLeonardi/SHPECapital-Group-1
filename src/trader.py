import argparse
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------- Helpers -----------------

def normalize_ohlcv(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """
    Return a single-index OHLCV with columns:
    ['Open','High','Low','Close','Adj Close','Volume'] when available.
    - Works with single-level or MultiIndex columns from yfinance.
    - If 'Adj Close' is missing (e.g., auto_adjust=True), synthesize it from 'Close'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    # If MultiIndex (e.g., columns like ('Close','AAPL')), select the ticker level if present.
    if isinstance(df.columns, pd.MultiIndex):
        # Try to slice by the ticker symbol on the last level.
        last_level = df.columns.levels[-1]
        if ticker and ticker in last_level:
            try:
                df = df.xs(ticker, axis=1, level=-1, drop_level=True)
            except Exception:
                # Fallback: flatten by taking first level name
                df.columns = [c[0] for c in df.columns]
        else:
            # Not a per-ticker MultiIndex; flatten by first level
            df.columns = [c[0] for c in df.columns]

    # Keep only known columns if they exist
    cols_order = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    present = [c for c in cols_order if c in df.columns]
    df = df[present].copy()

    # If Adj Close missing (because auto_adjust=True), synthesize it from Close
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    # Ensure standard order where available
    present = [c for c in cols_order if c in df.columns]
    df = df[present]

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift()
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative sign(volume * price_change)."""
    direction = np.sign(close.diff().fillna(0))
    return (volume.fillna(0) * direction).cumsum().fillna(0)


def ema_slope(series: pd.Series, span: int, lookback: int = 5) -> pd.Series:
    e = ema(series, span)
    return (e - e.shift(lookback)) / (lookback * series.replace(0, np.nan))


def roc(series: pd.Series, window: int = 10) -> pd.Series:
    return series.pct_change(window).fillna(0)


def stoch_rsi(series: pd.Series, rsi_window: int = 14, stoch_window: int = 14) -> pd.Series:
    r = rsi(series, rsi_window)
    rmin = r.rolling(stoch_window).min()
    rmax = r.rolling(stoch_window).max()
    denom = (rmax - rmin).replace(0, np.nan)
    out = (r - rmin) / denom
    return (out.clip(0, 1).fillna(0) * 100)


def vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume-Price Trend: cumulative(volume * pct_change(close))."""
    return (volume.fillna(0) * close.pct_change().fillna(0)).cumsum()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm.fillna(0) * volume.fillna(0)
    denom = volume.rolling(window).sum().replace(0, np.nan)
    return (mfv.rolling(window).sum() / denom).fillna(0)


def bb_width(close: pd.Series, window: int = 20, nstd: float = 2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + nstd * std
    lower = mid - nstd * std
    denom = mid.replace(0, np.nan)
    return ((upper - lower) / denom).fillna(0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder’s ADX implementation."""
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)


# ------------ Data helpers ------------

SECTOR_ETF_MAP = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK',
    'AMZN': 'XLY', 'META': 'XLC', 'GOOGL': 'XLC',
    'JPM': 'XLF',  'BAC':  'XLF', 'C':    'XLF',
    'XOM': 'XLE',  'CVX':  'XLE',
    # fallback to SPY if not found
}

def fetch_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, group_by='column')
    return normalize_ohlcv(df, ticker=ticker)


# ------------ Feature builders ------------

def market_component(start: str, end: str) -> pd.Series:
    """Breadth proxy + VIX trend → map to [-1, 1]."""
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, progress=False, group_by='column')
    spx = normalize_ohlcv(spx, ticker="^GSPC")
    if spx.empty or 'Close' not in spx.columns:
        return None

    spx['ema50'] = ema(spx['Close'], 50)
    spx['ema200'] = ema(spx['Close'], 200)

    # crude breadth proxy: fraction of recent days above EMAs
    above50 = (spx['Close'] > spx['ema50']).rolling(50, min_periods=20).mean()
    above200 = (spx['Close'] > spx['ema200']).rolling(50, min_periods=20).mean()
    breadth = 0.4 * above50 + 0.6 * above200  # 0..1

    # VIX trend (falling = risk-on)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False, group_by='column')
    vix = normalize_ohlcv(vix, ticker="^VIX")
    vix_ok = pd.Series(0.0, index=spx.index)
    if not vix.empty and 'Close' in vix.columns:
        vix50 = ema(vix['Close'], 50).reindex(spx.index).ffill()
        vix_ok = (vix['Close'].reindex(spx.index).ffill() < vix50).astype(float)

    mkt = 2 * (0.8 * breadth + 0.2 * vix_ok).clip(0, 1) - 1  # -> [-1, 1]
    mkt.name = 'mkt_comp'
    return mkt


def compute_signals(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    Build component scores in [-1, 1]:
      trend_comp, mom_comp, vol_comp, vola_comp
    Leaves original OHLCV columns intact and adds helper columns as needed.
    """
    df = df.copy()

    # ---- Trend (EMA50/200, EMA10/20, slope, ADX, sector confirm) ----
    df['ema10'] = ema(df['Close'], 10)
    df['ema20'] = ema(df['Close'], 20)
    df['ema50'] = ema(df['Close'], 50)
    df['ema200'] = ema(df['Close'], 200)

    slope200 = ema_slope(df['Close'], 200, lookback=5).clip(-0.02, 0.02) / 0.02   # ~[-1,1]
    adx14 = (adx(df['High'], df['Low'], df['Close'], 14) / 50.0).clip(0, 1)       # ~0..2 -> 0..1

    short_align = (((df['ema10'] > df['ema20']).astype(int) +
                    (df['ema50'] > df['ema200']).astype(int)) / 2.0)               # 0..1
    long_up = (df['Close'] > df['ema200']).astype(int)                             # 0/1

    # Sector/ETF confirmation
    sec = SECTOR_ETF_MAP.get(str(ticker).upper(), 'SPY')
    sec_df = fetch_price(
        sec,
        df.index.min().strftime('%Y-%m-%d'),
        df.index.max().strftime('%Y-%m-%d')
    )

    sec_trend = pd.Series(0.0, index=df.index)
    if not sec_df.empty and 'Close' in sec_df.columns:
        sec_df['ema200'] = ema(sec_df['Close'], 200)
        sec_trend = (sec_df['Close'] > sec_df['ema200']).reindex(df.index).ffill().astype(float)  # 0/1

    trend_raw = (
        0.6 * long_up +
        0.2 * short_align +
        0.1 * adx14 +
        0.1 * (slope200.clip(-1, 1) + 1) / 2 +
        0.1 * sec_trend
    )
    df['trend_comp'] = (2 * trend_raw.clip(0, 1) - 1).fillna(0)  # -> [-1,1]

    # ---- Momentum (RSI w/ dynamic threshold, StochRSI, ROC, simple divergence nudge) ----
    df['rsi14'] = rsi(df['Close'], 14)
    trend_bias = (df['rsi14'] > 55).astype(int)
    rsi_norm = np.where(trend_bias, (df['rsi14'] - 40) / 40, (df['rsi14'] - 30) / 40)
    rsi_norm = pd.Series(rsi_norm, index=df.index).clip(0, 1)

    st_rsi = (stoch_rsi(df['Close'], 14, 14) / 100.0).clip(0, 1)
    rroc = (roc(df['Close'], 10) / 0.05 + 0.5).clip(0, 1)  # -5%..+5% -> 0..1

    # simple divergence hint: price LL with RSI higher low => +, price HH with RSI lower high => -
    ll = df['Close'].rolling(60).apply(lambda x: 1.0 if np.argmin(x) == len(x) - 1 else 0.0, raw=False).fillna(0)
    hh = df['Close'].rolling(60).apply(lambda x: 1.0 if np.argmax(x) == len(x) - 1 else 0.0, raw=False).fillna(0)
    rsi_min_shift = df['rsi14'].rolling(60).min().shift(30)
    rsi_max_shift = df['rsi14'].rolling(60).max().shift(30)
    bull_div = ((ll == 1) & (df['rsi14'] > rsi_min_shift)).astype(float)
    bear_div = ((hh == 1) & (df['rsi14'] < rsi_max_shift)).astype(float)
    div_nudge = 0.1 * (bull_div - bear_div)  # small effect

    mom_raw = 0.5 * rsi_norm + 0.3 * st_rsi + 0.2 * rroc + div_nudge
    df['mom_comp'] = (2 * mom_raw.clip(0, 1) - 1).fillna(0)

    # ---- Volume (OBV EMA, VPT, CMF, 20D spike) ----
    df['obv'] = obv(df['Close'], df['Volume'])
    obv_ema10 = df['obv'].ewm(span=10, adjust=False).mean()
    vpt_s = vpt(df['Close'], df['Volume'])
    cmf20 = cmf(df['High'], df['Low'], df['Close'], df['Volume'], 20)

    # z-score helpers
    def zc(s: pd.Series, w: int = 100) -> pd.Series:
        m = s.rolling(w, min_periods=20).mean()
        sd = s.rolling(w, min_periods=20).std().replace(0, np.nan)
        return ((s - m) / sd).clip(-1, 1).fillna(0)

    volnorm = (df['Volume'] / df['Volume'].rolling(20).mean()).replace([np.inf, -np.inf], np.nan).fillna(0)
    spike = ((volnorm - 1.0) / 2.0).clip(0, 1)  # >1 means above avg; >3 big spike
    vol_raw = 0.5 * zc(obv_ema10) + 0.3 * zc(vpt_s) + 0.2 * (cmf20.clip(-1, 1)) + 0.1 * (2 * spike - 1)
    df['vol_comp'] = vol_raw.clip(-1, 1).fillna(0)

    # ---- Volatility (ATR% + BB width compression + “expansion from low base”) ----
    df['atr14'] = atr(df['High'], df['Low'], df['Close'], 14)
    df['atr_pct'] = (df['atr14'] / df['Close'].replace(0, np.nan)).fillna(0)
    width = bb_width(df['Close'], 20, 2.0).clip(0, 0.2)

    # prefer compression; map to 0..1 then -> [-1,1]
    comp = (0.06 - width).clip(-0.04, 0.06) / 0.06
    atr_pref = (0.03 - df['atr_pct']).clip(-0.02, 0.03) / 0.03
    med = df['atr_pct'].rolling(50, min_periods=20).median()
    expand = (df['atr_pct'] > 1.05 * med).astype(float)  # 5% above rolling median

    vola_raw = 0.6 * comp + 0.3 * atr_pref + 0.1 * expand
    df['vola_comp'] = (2 * vola_raw.clip(0, 1) - 1).fillna(0)

    return df


def weighted_score(df: pd.DataFrame, mkt_comp: pd.Series | None) -> pd.Series:
    """
    Combine component scores into a single score in [-1, 1],
    preserving your existing backtest interface.
    """
    mc = (mkt_comp.reindex(df.index).ffill().fillna(0)) if mkt_comp is not None \
         else pd.Series(0.0, index=df.index, name='mkt_comp')

    w_trend = 0.30
    w_mom   = 0.20
    w_vol   = 0.20
    w_vola  = 0.15
    w_mkt   = 0.15

    score = (
        w_trend * df['trend_comp'] +
        w_mom   * df['mom_comp']   +
        w_vol   * df['vol_comp']   +
        w_vola  * df['vola_comp']  +
        w_mkt   * mc
    )
    score.name = 'score'
    return score

def backtest(df, score, entry_thr=0.2, exit_thr=0.0, fee_bps=2, slippage_bps=0):
 
    df = df.copy()
    df['score'] = score
    # Signals
    df['signal_raw'] = np.nan
    df.loc[df['score'] > entry_thr, 'signal_raw'] = 1.0   # enter/keep long
    df.loc[df['score'] < exit_thr,  'signal_raw'] = 0.0   # exit to flat
    df['position'] = df['signal_raw'].ffill().fillna(0.0)
    #Last state
    df['position'] = df['signal_raw'].replace(to_replace=0, method='ffill').fillna(0)

    # Daily returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # Trading costs when position changes
    trade = df['position'].diff().abs().fillna(0)
    cost = trade * ((fee_bps + slippage_bps) / 10000.0)

    # Strategy returns
    df['strat_ret'] = df['position'].shift(1).fillna(0) * df['ret'] - cost
    df['equity'] = (1 + df['strat_ret']).cumprod()

    # Metrics
    daily = df['strat_ret']
    ann_factor = 252
    cagr = df['equity'].iloc[-1] ** (ann_factor / len(df)) - 1 if len(df) > 0 else 0
    sharpe = (daily.mean() / (daily.std() + 1e-9)) * np.sqrt(ann_factor) if daily.std() > 0 else 0
    cum = (1 + daily).cumprod()
    peak = cum.cummax()
    mdd = (cum / peak - 1).min()

    # Win rate
    wins = (daily > 0).sum()
    wr = wins / max(1, (daily != 0).sum())

    summary = {
        'CAGR': cagr,
        'Sharpe(252)': sharpe,
        'MaxDD': mdd,
        'WinRate': wr,
        'Trades': int(trade.sum())
    }
    return df, summary

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True, help='Ticker symbol, e.g., AAPL')
    ap.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    ap.add_argument('--end', default=dt.date.today().strftime('%Y-%m-%d'), help='End date YYYY-MM-DD (default: today)')
    ap.add_argument('--fee_bps', type=float, default=2.0, help='Round-trip fee in basis points')
    ap.add_argument('--slip_bps', type=float, default=1.0, help='Slippage in basis points')
    args = ap.parse_args()

    df = fetch_price(args.ticker, args.start, args.end)
    if df.empty:
        print(f"[!] No data for {args.ticker} between {args.start} and {args.end}")
        return

    df = compute_signals(df, ticker=args.ticker)
    mkt_comp = market_component(args.start, args.end)

    score = weighted_score(df, mkt_comp)

    bt, summary = backtest(
        df,
        score,
        entry_thr=0.2,
        exit_thr=0.0,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps
    )

    print(summary)

if __name__ == "__main__":
    main()