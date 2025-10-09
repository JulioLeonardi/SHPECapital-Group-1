from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd
import yfinance as yf


CACHE_DIR = Path("data/cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)




def _cache_path(tickers: Iterable[str], start: str, end: str, interval: str) -> Path:
    key = f"{','.join(sorted(tickers))}_{start}_{end}_{interval}.parquet".replace("/", "-")
    return CACHE_DIR / key




def get_prices(tickers: list[str], start: str, end: str, interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
    """Download OHLCV for tickers via yfinance.
    Returns MultiIndex (ticker, date) with columns: Open, High, Low, Close, Adj Close, Volume.
    """
    cache_file = _cache_path(tickers, start, end, interval)
    if cache_file.exists() and not force_refresh:
        return pd.read_parquet(cache_file)


    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )
    frames = []
    for t in tickers:
        if t not in raw:
            continue
        df = raw[t].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df["ticker"] = t
        df = df.rename_axis("date").reset_index()
        frames.append(df)
    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.set_index(["ticker", "date"]).sort_index()
    out.columns = [c.strip().replace(" ", "_") for c in out.columns]
    out.to_parquet(cache_file)
    return out