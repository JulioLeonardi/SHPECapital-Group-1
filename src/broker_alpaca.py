import os
from typing import Optional
import alpaca_trade_api as tradeapi


def _get_keys():
    """
    Try Alpaca's standard env vars first (APCA_*),
    then fall back to ALPACA_* if you used those instead.
    """
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL")
    
    return key, secret, base_url


def get_client() -> tradeapi.REST:
    """
    Returns a REST client connected to the paper endpoint.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in env.
    """
    key, secret, base_url = _get_keys()
    return tradeapi.REST(key, secret, base_url, api_version="v2")


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


def submit_flatten_order(
    client: tradeapi.REST,
    symbol: str,
    qty: int,
    side: str,
    time_in_force: str = "day",
):
    """
    Simple market order (no bracket). Used to flatten an existing position.
    side: 'sell' or 'buy' depending on direction.
    """
    if qty <= 0:
        return

    print(f"[Alpaca] {side.upper()} market order: {qty} {symbol}")
    client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force=time_in_force,
    )


def submit_bracket_entry(
    client: tradeapi.REST,
    symbol: str,
    qty: int,
    side: str,
    stop_price: float,
    take_profit_price: Optional[float] = None,
    time_in_force: str = "gtc",
):
    """
    Submit a bracket order (entry + attached stop-loss (+ optional take-profit)).

    For a long trade:
      - side = 'buy'
      - stop_price < current price
      - take_profit_price > current price (optional)
    """
    if qty <= 0:
        return

    stop_loss = {"stop_price": f"{stop_price:.2f}"}
    take_profit = None
    if take_profit_price is not None:
        take_profit = {"limit_price": f"{take_profit_price:.2f}"}

    print(
        f"[Alpaca] BRACKET {side.upper()} {qty} {symbol} "
        f"stop={stop_loss['stop_price']}, "
        f"tp={take_profit['limit_price'] if take_profit else 'None'}"
    )

    client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force=time_in_force,
        order_class="bracket",
        stop_loss=stop_loss,
        take_profit=take_profit,
    )