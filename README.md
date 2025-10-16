SHPE CAPITAL GROUP 1

Technical Indicators:

 1. Trend Indicator 
 - Use EMA (Exponential Moving Average) 50/200
 - Buy when 50EMA > 200EMA
 - Sell when 50EMA < 200EMA
 - Weight: 30%

 2. Momentum Indicator
 - RSI (Relative Strength Index) (14 day timeframe)
 - high -> buy, low -> sell
 - Weight: 20%
 
 3. Volume Indicator
 - OBV (On-Balance Volume)
 - rising -> buy, falling -> sell
 - Weight: 20%

 4. Volatility Indicator
 - ATR (Average True Range) (14 day timeframe)
 - high -> buy, low -> sell
 - Weight: 15%

 5. Market Indicator
 - % of companies on S&P above 200EMA
 - high -> buy, low -> sell
 - Weight: 15%

How to get this working:

 1) Set up Python
 - Install Python 3.10+ from python.org.
 - Open **PowerShell** (Windows) or **Terminal** (mac/Linux).

 2) Create & activate a virtual environment
 **Windows (PowerShell)**
 python -m venv .venv
 .\.venv\Scripts\Activate.ps1
 # If blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

 **MacOS and Linux**
 python3 -m venv .venv
 source .venv/bin/activate

 3) Install dependencies
 pip install -r requirements.txt

 4) Backtest
 python -m src.trader --ticker AAPL --start 2015-01-01 --fee_bps 2