import yfinance as yf
import pandas as pd

def normalize_ohlcv(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """
    Return a single-index OHLCV with columns:
    ['Open','High','Low','Close','Adj Close','Volume'] when available.
    Works with yfinance's single or MultiIndex columns.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    # Handle MultiIndex columns (e.g., ('Close','AAPL'))
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

    # If Adj Close missing, synthesize it from Close
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    present = [c for c in cols_order if c in df.columns]
    df = df[present]

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

SECTOR_ETF_MAP = {
    # extend as needed
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'AMZN': 'XLY', 'META': 'XLC', 'GOOGL': 'XLC',
    'JPM': 'XLF', 'BAC': 'XLF', 'C': 'XLF','XOM': 'XLE', 'CVX': 'XLE',
}

def fetch_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, group_by='column')
    return normalize_ohlcv(df, ticker=ticker)

def fetch_refs(ticker: str, start: str, end: str):
    market = fetch_price('SPY', start, end)
    vix = fetch_price('^VIX', start, end)
    sector = fetch_price(SECTOR_ETF_MAP.get(ticker.upper(), 'SPY'), start, end)
    return market, vix, sector