import yfinance as yf  

# Fetch historical data for AAPL
df = yf.download("AAPL", start="2025-01-01", end="2025-02-01", auto_adjust=True)

# Check if data was successfully retrieved
if df.empty:
    print("Failed to fetch data. Please check ticker or date range.")
else:
    print(df.head())  # Display first few rows