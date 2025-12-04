# SHPE CAPITAL GROUP 1

## 1. Strategy Overview

The strategy is a weighted factor model built from five pillars:

1. **Trend (30%)**
   - EMAs (10/20/50/200), long-term slope, and ADX for trend strength.
   - Optional confirmation from the stock’s **sector ETF** (e.g., AAPL → XLK).
   - Outputs a 0–1 **trend score** (higher = stronger uptrend with sector confirmation).

2. **Momentum (20%)**
   - RSI(14), StochRSI, 10-day ROC, and simple bullish/bearish RSI divergences.
   - Dynamic RSI thresholds (more forgiving when trend is strong).
   - 0–1 **momentum score** (higher = cleaner upside momentum).

3. **Volume / Flow (20%)**
   - OBV(EMA-smoothed), Volume-Price Trend (VPT), Chaikin Money Flow, and volume spikes vs 20-day average.
   - Captures whether price moves are backed by **real money flow**.
   - 0–1 **volume score**.

4. **Volatility / Setup Quality (15%)**
   - ATR% (ATR / Close), Bollinger Band width, and “volatility expansion from low base.”
   - Prefers **compressed / moderate volatility** that is starting to expand (classic breakout behavior).
   - 0–1 **volatility score** (higher = cleaner volatility regime).

5. **Market Breadth & Risk Environment (15%)**
   - SPY as market proxy: % of days above its 50/200-day EMA.
   - Optional overlay from VIX trend (falling VIX = risk-on).
   - 0–1 **breadth score** (higher = healthier backdrop).

The factor scores are then combined into a single composite score. 
If the score is above a threshold, we enter long. If it is below another threshold, we exit flat

The size of our position is determined using ATR% scores to keep risk of trade constant 
(raw_size) = min(max_leverage, risk_per_trade / ATR%) 

## 2. How to get this working:

1) **Set up Python**
   - Install Python 3.10+ from python.org.
   - Open **PowerShell** (Windows) or **Terminal** (mac/Linux).

2) **Create & activate a virtual environment**
   **Windows (PowerShell)**
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   # If blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

   **MacOS and Linux**
   python3 -m venv .venv
   source .venv/bin/activate

3) **Install dependencies**
   pip install -r requirements.txt

4) **Backtest**
   Example:

   Inside src (cd src)
   python -m backtest --ticker AAPL --start 2015-01-01 --fee_bps 2

   Arguments:

   --ticker – Symbol (default: AAPL).
   --start – Start date (YYYY-MM-DD).
   --end – Optional end date (defaults to today).
   --fee_bps – One-way trading fee in basis points (default 2 bps).
   --slip_bps – Additional slippage in bps (default 0).

   Running the script will: 
   - Download data for the target ticker
   - Download SPY, ^VIX, and the sector ETF as references
   - Build factor scores
   - Run a simple daily-rebalanced backtest with trading costs included
   - Prints a summary to the terminal

5) **Trading with Alpaca**

   Right now we are using this with paper trading. You would have to setup your
   .env in the root directory with your ALPACA_API_KEY, ALPACA_SECRET_KEY, and
   ALPACA_BASE_URL.

   After that you can run a rebalance cycle using:

   python live_trader.py --ticker AAPL 2025-01-01

   The script uses our same composite scores to determine the most optimal transaction
   with the most up to date information at the time it was done. 