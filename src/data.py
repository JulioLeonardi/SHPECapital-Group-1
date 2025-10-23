import yfinance as yf
import pandas as pd

SECTOR_ETF_MAP = {
    'AAPL': 'XLK', 
    'MSFT': 'XLK',
    'XOM':  'XLE',
    'JPM':  'XLF',
}

def fetch(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False)
    df = df[['Open','High','Low','Close','Adj Close','Volume']].dropna()
    return df

def fetch_refs(ticker: str, start: str):
    market = fetch('SPY', start)
    vix    = fetch('^VIX', start)
    sector = fetch(SECTOR_ETF_MAP.get(ticker, 'SPY'), start)
    return market, vix, sector
