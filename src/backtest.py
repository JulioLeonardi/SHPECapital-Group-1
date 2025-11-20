import argparse
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from config import StrategyConfig
from data import fetch_price as fetch_price_mod, fetch_refs
from strategy import build_strategy_signals

# ------------ Utilities ------------

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
        last_level = df.columns.levels[-1]
        if ticker and ticker in last_level:
            try:
                df = df.xs(ticker, axis=1, level=-1, drop_level=True)
            except Exception:
                df.columns = [c[0] for c in df.columns]
        else:
            df.columns = [c[0] for c in df.columns]

    cols_order = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    present = [c for c in cols_order if c in df.columns]
    df = df[present].copy()

    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    present = [c for c in cols_order if c in df.columns]
    df = df[present]

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def backtest(df: pd.DataFrame, fee_bps: float = 2.0, slippage_bps: float = 0.0):
    """
    Backtest assuming:
      - df['Signal'] in {0,1}
      - df['PosSize'] in [0, max_leverage], e.g. fraction of equity
    Position = Signal * PosSize (fraction of equity).
    """
    df = df.copy()

    if 'Signal' not in df.columns or 'PosSize' not in df.columns:
        raise ValueError("DataFrame must contain 'Signal' and 'PosSize' columns for this backtest.")

    # Daily underlying returns
    df['ret'] = df['Close'].pct_change().fillna(0)

    # Position as fraction of equity
    df['position'] = (df['Signal'] * df['PosSize']).fillna(0.0)
    df['position'] = df['position'].ffill().fillna(0.0)

    # Turnover and trading costs
    trade = df['position'].diff().abs().fillna(0.0)
    cost = trade * ((fee_bps + slippage_bps) / 10000.0)

    # Strategy returns: previous day's position * next day's return - costs
    df['strat_ret'] = df['position'].shift(1).fillna(0.0) * df['ret'] - cost
    df['equity'] = (1 + df['strat_ret']).cumprod()

    # Metrics
    daily = df['strat_ret']
    ann_factor = 252

    if len(df) > 0:
        cagr = df['equity'].iloc[-1] ** (ann_factor / len(df)) - 1
    else:
        cagr = 0.0

    if daily.std() > 0:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(ann_factor)
    else:
        sharpe = 0.0

    cum = (1 + daily).cumprod()
    peak = cum.cummax()
    mdd = (cum / peak - 1).min()

    wins = (daily > 0).sum()
    wr = wins / max(1, (daily != 0).sum())

    summary = {
        "CAGR": cagr,
        "Sharpe(252)": sharpe,
        "MaxDD": mdd,
        "WinRate": wr,
        "Trades": int(trade.sum())
    }
    return df, summary

# ------------ Data helpers ------------

SECTOR_ETF_MAP = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK',
    'AMZN': 'XLY', 'META': 'XLC', 'GOOGL': 'XLC',
    'JPM': 'XLF',  'BAC':  'XLF', 'C':    'XLF',
    'XOM': 'XLE',  'CVX':  'XLE',
}

def fetch_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, group_by='column')
    return normalize_ohlcv(df, ticker=ticker)

def fetch_refs(ticker: str, start: str, end: str):
    market = fetch_price('SPY', start, end)
    vix = fetch_price('^VIX', start, end)
    sector = fetch_price(SECTOR_ETF_MAP.get(ticker.upper(), 'SPY'), start, end)
    return market, vix, sector

# ------------ CLI entrypoint ------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=dt.date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--fee_bps", type=float, default=2)
    ap.add_argument("--slip_bps", type=float, default=0)
    args = ap.parse_args()

    # --- Fetch target price series ---
    df = fetch_price_mod(args.ticker, args.start, args.end)
    if df.empty:
        print(f"[!] No data for {args.ticker} between {args.start} and {args.end}")
        return

    # --- Market, sector, VIX references ---
    market_df, vix_df, sector_df = fetch_refs(args.ticker, args.start, args.end)

    # --- Build signals via unified modular pipeline ---
    cfg = StrategyConfig()
    sigs = build_strategy_signals(
        tdf=df,
        market_df=market_df,
        sector_df=sector_df,
        vix_df=vix_df,
        cfg=cfg
    )

    # Merge with tdf for backtest
    merged = df.join(sigs, how="inner")

    # Backtest using your existing code
    bt, summary = backtest(
        merged,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps
    )

    print(summary)

if __name__ == "__main__":
    main()