import argparse
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------- Helpers -----------------
def normalize_ohlcv(df, ticker=None):
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        if ticker is not None:
            try:
                df = df.xs(ticker, level=1, axis=1)
            except Exception:
                if df.columns.nlevels > 1 and len(df.columns.get_level_values(-1).unique()) == 1:
                    df.columns = df.columns.get_level_values(0)
        else:
            if df.columns.nlevels > 1 and len(df.columns.get_level_values(-1).unique()) == 1:
                df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns and hasattr(df[col], 'ndim') and getattr(df[col], 'ndim', 1) != 1:
            df[col] = df[col].iloc[:, 0]

    return df

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def market_component(start, end):

    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False, group_by='column')
    spx = normalize_ohlcv(spx, ticker="^GSPC")
    if spx.empty or 'Close' not in spx.columns:
        return None

    spx['ema200'] = ema(spx['Close'], 200)

    close_s = spx['Close']
    ema200_s = spx['ema200']

    comp = pd.Series(np.where(close_s > ema200_s, 1.0, -1.0),
                     index=spx.index, name='mkt_comp')
    return comp

def compute_signals(df):
    # Trend
    df['ema50'] = ema(df['Close'], 50)
    df['ema200'] = ema(df['Close'], 200)
    df['trend_comp'] = np.where(df['ema50'] > df['ema200'], 1.0, -1.0)

    # Momentum (RSI)
    df['rsi14'] = rsi(df['Close'], 14)
    df['mom_comp'] = ((df['rsi14'] - 50.0) / 50.0).clip(-1, 1)

    # Volume (OBV)
    df['obv'] = obv(df['Close'], df['Volume'])
    df['obv_slope'] = df['obv'].diff(5)
    df['vol_comp'] = np.sign(df['obv_slope']).fillna(0)

    # Volatility (ATR)
    assert df['High'].ndim == df['Low'].ndim == 1

    df['atr14'] = atr(df['High'], df['Low'], df['Close'], 14)

    close_1d = np.asarray(df['Close'], dtype='float64').reshape(-1)
    atr14_1d = np.asarray(df['atr14'], dtype='float64').reshape(-1)

    atr_pct_arr = np.divide(
        atr14_1d, close_1d,
        out=np.full_like(atr14_1d, np.nan),
        where=np.isfinite(close_1d) & (close_1d != 0)
    )

    df['atr_pct'] = pd.Series(atr_pct_arr, index=df.index, name='atr_pct')

    mean100 = df['atr_pct'].rolling(100, min_periods=20).mean()
    std100  = df['atr_pct'].rolling(100, min_periods=20).std()
    z = (df['atr_pct'] - mean100) / (std100.replace(0, np.nan))
    df['vola_comp'] = z.clip(-1, 1).fillna(0)

    return df

def weighted_score(df, mkt_comp):

    mc = (mkt_comp.reindex(df.index).ffill().fillna(0)) if mkt_comp is not None else pd.Series(0.0, index=df.index, name='mkt_comp')

    # Weights
    w_trend = 0.30
    w_mom = 0.20
    w_vol = 0.20
    w_vola = 0.15
    w_mkt = 0.15

    score = (
        w_trend * df['trend_comp'] +
        w_mom   * df['mom_comp'] +
        w_vol   * df['vol_comp'] +
        w_vola  * df['vola_comp'] +
        w_mkt   * mc
    )
    return score.rename('score')

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
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', type=str, default='AAPL')
    p.add_argument('--start', type=str, default='2015-01-01')
    p.add_argument('--end', type=str, default=None)
    p.add_argument('--fee_bps', type=float, default=2.0)
    p.add_argument('--slip_bps', type=float, default=0.0)
    args = p.parse_args()

    df = yf.download(
        args.ticker,
        start=args.start,
        end=args.end,
        auto_adjust=True,
        progress=False,
        group_by='column'
    )
    df = normalize_ohlcv(df, ticker=args.ticker)

    if df.empty or not set(['High', 'Low', 'Close', 'Volume']).issubset(df.columns):
        raise RuntimeError("Downloaded data missing required OHLCV columns.")

    df = compute_signals(df)

    mkt_comp = market_component(args.start, args.end)

    score = weighted_score(df, mkt_comp)
    bt, summary = backtest(
        df, score,
        entry_thr=0.2,
        exit_thr=0.0,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps
    )

    print(f"=== {args.ticker} Backtest ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"Final equity: {bt['equity'].iloc[-1]:.4f}")

if __name__ == "__main__":
    main()
