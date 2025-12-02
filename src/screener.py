import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

# Import from existing modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data import normalize_ohlcv
from strategy import build_strategy_signals
from config import StrategyConfig


# ----------------- Universe Definition -----------------
def get_sp500_tickers():
    """Get S&P 500 ticker list from Wikipedia."""
    try:
        # Handle SSL certificate issues and add User-Agent to avoid 403 errors
        import ssl
        import urllib.request

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create request with User-Agent header
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, context=ssl_context) as response:
            tables = pd.read_html(response.read())
            # The S&P 500 constituent table is the second table (index 1)
            sp500 = tables[1]

        tickers = sp500['Symbol'].str.replace('.', '-').tolist()
        print(f"✓ Retrieved {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        return []


def get_russell1000_subset():
    """Alternative: Use a predefined list of large-cap tickers."""
    # Top ~100 liquid stocks as fallback
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'V', 'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'UNH', 'DIS', 'HD', 'BAC',
        'ADBE', 'CRM', 'NFLX', 'CMCSA', 'XOM', 'NKE', 'PFE', 'ABBV', 'TMO',
        'COST', 'AVGO', 'CVX', 'MRK', 'CSCO', 'ACN', 'AMD', 'LLY', 'PEP',
        'TXN', 'QCOM', 'NEE', 'UNP', 'DHR', 'INTC', 'PM', 'HON', 'RTX',
        'LOW', 'UPS', 'ORCL', 'INTU', 'IBM', 'SBUX', 'CAT', 'GS', 'BA',
        'AMGN', 'ISRG', 'CVS', 'BLK', 'SPGI', 'DE', 'GE', 'AXP', 'MMM',
        'GILD', 'BKNG', 'MDLZ', 'TGT', 'MO', 'SYK', 'TMUS', 'ZTS', 'CI',
        'PLD', 'LRCX', 'CB', 'REGN', 'SCHW', 'ADI', 'MU', 'DUK', 'SO',
        'EQIX', 'CL', 'ATVI', 'FISV', 'ITW', 'BSX', 'AON', 'APD', 'SHW',
        'TJX', 'MMC', 'ICE', 'USB', 'PNC', 'WM', 'GD', 'HUM', 'NSC', 'FDX'
    ]
    return tickers


# ----------------- Stage 1: Fundamental Screen -----------------
def quick_screen(tickers, min_market_cap=2e9, min_volume=1e6, max_price=None, verbose=True):
    """
    Quick fundamental screen using only current metadata (no historical data).

    Args:
        tickers: List of ticker symbols
        min_market_cap: Minimum market cap in dollars (default $2B)
        min_volume: Minimum average volume (default 1M shares)
        max_price: Maximum stock price to filter out very expensive stocks
        verbose: Print progress

    Returns:
        List of tickers that pass the screen
    """
    candidates = []
    failed = []

    print(f"\n{'='*60}")
    print(f"STAGE 1: FUNDAMENTAL SCREEN")
    print(f"{'='*60}")
    print(f"Screening {len(tickers)} tickers...")
    print(f"Filters: Tech Sector, Market Cap >${min_market_cap/1e9:.1f}B, Volume >{min_volume/1e6:.1f}M")

    for i, ticker in enumerate(tickers):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(tickers)} ({len(candidates)} passed)")

        try:
            info = yf.Ticker(ticker).info

            # Tech sector filter
            sector = info.get('sector', '')
            if sector not in ['Technology', 'Communication Services']:
                continue

            market_cap = info.get('marketCap', 0)
            avg_volume = info.get('averageVolume', 0)
            price = info.get('currentPrice', info.get('regularMarketPrice', 0))

            # Apply filters
            if market_cap < min_market_cap:
                continue
            if avg_volume < min_volume:
                continue
            if max_price and price > max_price:
                continue

            candidates.append(ticker)

        except Exception as e:
            failed.append(ticker)
            continue

    print(f"\n✓ Stage 1 complete: {len(candidates)}/{len(tickers)} passed")
    if failed and verbose:
        print(f"  ({len(failed)} tickers failed to fetch)")

    return candidates


# ----------------- Stage 2: Technical Screen -----------------
def technical_screen(tickers, lookback='6mo', verbose=True):
    """
    Screen using recent price data and basic technical indicators.

    Filters:
        - Price above 50-day moving average (uptrend)
        - Positive 3-month return (momentum)
        - Daily volatility between 2-8% (growth stocks)
        - Not in severe drawdown (max -35%)

    Args:
        tickers: List of ticker symbols
        lookback: How much recent data to download ('6mo', '1y', etc.)
        verbose: Print progress

    Returns:
        List of tickers that pass technical filters
    """
    shortlist = []

    print(f"\n{'='*60}")
    print(f"STAGE 2: TECHNICAL SCREEN ({lookback} data)")
    print(f"{'='*60}")
    print(f"Downloading data for {len(tickers)} tickers...")

    try:
        # Download all at once (much faster)
        data = yf.download(
            tickers,
            period=lookback,
            group_by='ticker',
            progress=False,
            threads=True,
            auto_adjust=True
        )

        print(f"Applying technical filters...")

        for ticker in tickers:
            try:
                # Handle single vs multi-ticker download format
                if len(tickers) == 1:
                    df = data.copy()
                else:
                    df = data[ticker].copy()

                if df.empty or 'Close' not in df.columns:
                    continue

                # Filter 1: Above 50-day MA
                df['ma50'] = df['Close'].rolling(50, min_periods=20).mean()
                if df['Close'].iloc[-1] < df['ma50'].iloc[-1]:
                    continue

                # Filter 2: Positive momentum (3-month return > 0%)
                if len(df) < 60:  # Need at least ~3 months of data
                    continue
                ret_3m = (df['Close'].iloc[-1] / df['Close'].iloc[-60]) - 1
                if ret_3m < 0:
                    continue

                # Filter 3: Growth stock volatility range (2-8%)
                daily_vol = df['Close'].pct_change().std()
                if daily_vol < 0.02 or daily_vol > 0.08:  # Want 2-8% daily volatility
                    continue

                # Filter 4: Not in severe drawdown
                peak = df['Close'].rolling(252, min_periods=60).max()
                drawdown = (df['Close'].iloc[-1] / peak.iloc[-1]) - 1
                if drawdown < -0.35:  # Skip if down >35% from peak
                    continue

                shortlist.append(ticker)

            except Exception as e:
                if verbose:
                    pass  # Silently skip bad tickers
                continue

        print(f"\n✓ Stage 2 complete: {len(shortlist)}/{len(tickers)} passed")

    except Exception as e:
        print(f"Error in technical screen: {e}")
        return []

    return shortlist


# ----------------- Stage 3: Full Signal Scoring -----------------
def score_all(tickers, start='2020-01-01', end=None, verbose=True):
    """
    Compute full technical signals and scores for shortlisted tickers.

    Args:
        tickers: List of ticker symbols
        start: Start date for full history
        end: End date (None = today)
        verbose: Print progress

    Returns:
        Dictionary of {ticker: latest_score}
    """
    scores = {}

    print(f"\n{'='*60}")
    print(f"STAGE 3: FULL SIGNAL COMPUTATION")
    print(f"{'='*60}")
    print(f"Computing signals for {len(tickers)} tickers from {start}...")

    # Create config
    cfg = StrategyConfig()

    # Fetch market data once (SPY, VIX)
    try:
        market_df = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)
        market_df = normalize_ohlcv(market_df, 'SPY')
        vix_df = yf.download('^VIX', start=start, end=end, auto_adjust=True, progress=False)
        vix_df = normalize_ohlcv(vix_df, '^VIX')
    except Exception as e:
        print(f"Error downloading market data: {e}")
        return {}

    # Download all tickers at once
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by='ticker',
            progress=False,
            threads=True,
            auto_adjust=True
        )

        for i, ticker in enumerate(tickers):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(tickers)}")

            try:
                # Handle single vs multi-ticker format
                if len(tickers) == 1:
                    df = data.copy()
                else:
                    df = data[ticker].copy()

                df = normalize_ohlcv(df, ticker)

                if df.empty or not {'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
                    continue

                # Get sector ETF (default to SPY if unknown)
                sector_map = {
                    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'AMD': 'XLK',
                    'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC'
                }
                sector_etf = sector_map.get(ticker, 'XLK')  # Default to tech ETF
                sector_df = yf.download(sector_etf, start=start, end=end, auto_adjust=True, progress=False)
                sector_df = normalize_ohlcv(sector_df, sector_etf)

                # Compute strategy signals
                signals_df = build_strategy_signals(df, market_df, sector_df, vix_df, cfg)

                # Get latest composite score
                if not signals_df.empty and 'Score' in signals_df.columns:
                    scores[ticker] = signals_df['Score'].iloc[-1]

            except Exception as e:
                if verbose:
                    print(f"  Error processing {ticker}: {e}")
                continue

        print(f"\n✓ Stage 3 complete: {len(scores)}/{len(tickers)} successfully scored")

    except Exception as e:
        print(f"Error downloading data: {e}")
        return {}

    return scores


# ----------------- Stage 4: Portfolio Selection -----------------
def build_portfolio(scores, min_score=0.2, verbose=True):
    """
    Select top N stocks by score for final portfolio.
    Interactive version - asks user how many stocks they want.

    Args:
        scores: Dictionary of {ticker: score}
        min_score: Minimum score threshold (default 0.2)
        verbose: Print results

    Returns:
        List of selected tickers
    """
    print(f"\n{'='*60}")
    print(f"STAGE 4: PORTFOLIO SELECTION")
    print(f"{'='*60}")

    # Filter by minimum score
    qualified = {t: s for t, s in scores.items() if s > min_score}

    print(f"Tickers above threshold ({min_score}): {len(qualified)}/{len(scores)}")

    if not qualified:
        print("WARNING: No tickers above score threshold!")
        return []

    # Sort by score
    sorted_tickers = sorted(qualified.items(), key=lambda x: x[1], reverse=True)

    # Ask user how many stocks they want
    print(f"\n{len(qualified)} stocks passed all filters.")
    while True:
        try:
            user_input = input(f"How many stocks do you want in your portfolio? (1-{len(qualified)}): ").strip()
            n = int(user_input)
            if 1 <= n <= len(qualified):
                break
            else:
                print(f"Please enter a number between 1 and {len(qualified)}")
        except ValueError:
            print("Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled by user")
            return []

    # Take top N
    portfolio = [(ticker, score) for ticker, score in sorted_tickers[:n]]

    print(f"\n{'='*60}")
    print(f"FINAL PORTFOLIO ({len(portfolio)} stocks)")
    print(f"{'='*60}")
    print(f"{'Rank':<6}{'Ticker':<10}{'Score':>10}")
    print(f"{'-'*30}")

    for i, (ticker, score) in enumerate(portfolio, 1):
        print(f"{i:<6}{ticker:<10}{score:>10.4f}")

    avg_score = np.mean([s for _, s in portfolio])
    print(f"\nAverage Score: {avg_score:.4f}")

    return [ticker for ticker, _ in portfolio]


# ----------------- Main Pipeline -----------------
def screen_universe(
    universe='sp500',
    min_market_cap=2e9,
    min_volume=1e6,
    lookback='6mo',
    start='2020-01-01',
    min_score=0.2,
    verbose=True
):
    """
    Complete screening pipeline from universe to final portfolio.
    User will be prompted to select how many stocks they want at the end.

    Args:
        universe: 'sp500' or 'large_cap'
        min_market_cap: Minimum market cap filter ($)
        min_volume: Minimum average volume filter
        lookback: Lookback period for technical screen
        start: Start date for full signal computation
        min_score: Minimum score threshold
        verbose: Print detailed progress

    Returns:
        List of selected portfolio tickers
    """
    start_time = time.time()

    print(f"\n{'#'*60}")
    print(f"# PORTFOLIO SCREENING PIPELINE")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    # Stage 0: Get universe
    print(f"\nUniverse: {universe}")
    if universe == 'sp500':
        initial_tickers = get_sp500_tickers()
    else:
        initial_tickers = get_russell1000_subset()

    if not initial_tickers:
        print("ERROR: Failed to get initial universe")
        return []

    # Stage 1: Fundamental screen
    candidates = quick_screen(
        initial_tickers,
        min_market_cap=min_market_cap,
        min_volume=min_volume,
        verbose=verbose
    )

    if not candidates:
        print("ERROR: No candidates passed fundamental screen")
        return []

    # Stage 2: Technical screen
    shortlist = technical_screen(candidates, lookback=lookback, verbose=verbose)

    if not shortlist:
        print("ERROR: No candidates passed technical screen")
        return []

    # Stage 3: Full scoring
    scores = score_all(shortlist, start=start, verbose=verbose)

    if not scores:
        print("ERROR: Failed to score any tickers")
        return []

    # Stage 4: Portfolio selection (interactive)
    portfolio = build_portfolio(
        scores,
        min_score=min_score,
        verbose=verbose
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Funnel: {len(initial_tickers)} → {len(candidates)} → {len(shortlist)} → {len(scores)} → {len(portfolio)}")

    return portfolio


# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage portfolio screening pipeline"
    )
    parser.add_argument(
        '--universe',
        type=str,
        default='sp500',
        choices=['sp500', 'large_cap'],
        help='Starting universe (default: sp500)'
    )
    parser.add_argument(
        '--min_mcap',
        type=float,
        default=2.0,
        help='Minimum market cap in billions (default: 2.0)'
    )
    parser.add_argument(
        '--min_volume',
        type=float,
        default=1.0,
        help='Minimum avg volume in millions (default: 1.0)'
    )
    parser.add_argument(
        '--lookback',
        type=str,
        default='6mo',
        help='Lookback period for technical screen (default: 6mo)'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2020-01-01',
        help='Start date for full signal computation (default: 2020-01-01)'
    )
    parser.add_argument(
        '--min_score',
        type=float,
        default=0.2,
        help='Minimum score threshold (default: 0.2)'
    )

    args = parser.parse_args()

    # Run pipeline (interactive - user selects portfolio size at the end)
    portfolio = screen_universe(
        universe=args.universe,
        min_market_cap=args.min_mcap * 1e9,
        min_volume=args.min_volume * 1e6,
        lookback=args.lookback,
        start=args.start,
        min_score=args.min_score,
        verbose=True
    )

    # Print final portfolio tickers
    if portfolio:
        print(f"\nSelected Tickers: {', '.join(portfolio)}")


if __name__ == "__main__":
    main()
