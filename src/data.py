# data.py
import yfinance as yf
import pandas as pd
from trader import normalize_ohlcv

SECTOR_ETF_MAP = {
    # extend as needed
    'AAPL': 'XLK',   # Technology (approx)
    'MSFT': 'XLK',
    'XOM':  'XLE',
    'JPM':  'XLF',
}

def fetch_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, group_by='column')
    return normalize_ohlcv(df, ticker=ticker)

def fetch_refs(ticker: str, start: str):
    market = fetch_price('SPY', start)
    vix    = fetch_price('^VIX', start)
    sector = fetch_price(SECTOR_ETF_MAP.get(ticker, 'SPY'), start)
    return market, vix, sector
