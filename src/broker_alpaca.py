import os
from typing import Optional
import alpaca_trade_api as tradeapi


ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def get_client() -> tradeapi.REST:
    """
    Returns a REST client connected to the paper endpoint.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in env.
    """
    key = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET_KEY"]
    return tradeapi.REST(key, secret, base_url=ALPACA_BASE_URL, api_version="v2")


def get_account_equity(client: tradeapi.REST) -> float:
    """Return current account equity as float."""
    account = client.get_account()
    return float(account.equity)


def get_position_qty(client: tradeapi.REST, symbol: str) -> int:
    """
    Return current share quantity for symbol (0 if flat).
    """
    try:
        pos = client.get_position(symbol)
        return int(float(pos.qty))
    except tradeapi.rest.APIError as e:
        # Alpaca raises an APIError when no position exists
        if "position does not exist" in str(e).lower():
            return 0
        raise


def submit_market_order(
    client: tradeapi.REST,
    symbol: str,
    side: str,
    qty: int,
    time_in_force: str = "day",
):
    """
    Fire a simple market order (no bracket logic yet).
    side: "buy" or "sell"
    """
    if qty <= 0:
        return

    print(f"[Alpaca] submitting {side} order: {qty} {symbol}")
    client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force=time_in_force,
    )
