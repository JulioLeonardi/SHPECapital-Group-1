import argparse
import datetime as dt

from config import StrategyConfig
from data import fetch_price, fetch_refs
from strategy import build_strategy_signals
from position import stop_loss_levels
from broker_alpaca import (get_client, get_account_equity, get_position_qty, submit_flatten_order, submit_bracket_entry)



def compute_bracket_levels(df):
    """
    Given a merged df with OHLC and factor columns, compute today's
    stop-loss and take-profit based on ATR%.

    Returns (stop_price, tp_price).
    """
    sl_series = stop_loss_levels(df, multiple=2.0)  # 2x ATR% below price
    close = df["Close"]
    stop_price = float(sl_series.iloc[-1])
    last_close = float(close.iloc[-1])

    # Simple 1:1 R:R take profit: distance from entry to stop * 1
    risk_per_share = last_close - stop_price
    if risk_per_share <= 0:
        # fallback: small 2% stop + 2% TP if ATR logic degenerates (e.g., tiny ATR)
        risk_per_share = 0.02 * last_close
        stop_price = last_close - risk_per_share

    tp_price = last_close + risk_per_share  # 1R
    return stop_price, tp_price


def run_once(ticker: str, start: str):
    """
    Single evaluation + rebalance step:
      - Pulls data
      - Builds Signal + PosSize
      - Rebalances Alpaca account with bracket order + ATR-based stops
    """
    end = dt.date.today().strftime("%Y-%m-%d")

    # Price + refs
    tdf = fetch_price(ticker, start, end)
    if tdf.empty:
        raise RuntimeError(f"No data for {ticker} between {start} and {end}")

    market_df, vix_df, sector_df = fetch_refs(ticker, start, end)

    # Strategy signals
    cfg = StrategyConfig()
    sigs = build_strategy_signals(
        tdf=tdf,
        market_df=market_df,
        sector_df=sector_df,
        vix_df=vix_df,
        cfg=cfg,
    )

    merged = tdf.join(sigs, how="inner")
    last = merged.iloc[-1]

    last_date = last.name
    last_close = float(last["Close"])
    signal = int(last["Signal"])
    pos_frac = float(last["PosSize"])  # fraction of equity (0..max_leverage)

    print(f"=== {ticker} @ {last_date.date()} ===")
    print(
        f"Close={last_close:.2f} | "
        f"Score={last['Score']:.3f} | "
        f"Signal={signal} | PosSize={pos_frac:.3f}"
    )

    # Connect to Alpaca
    client = get_client()
    equity = get_account_equity(client)
    print(f"[Alpaca] Equity: {equity:.2f} USD")

    current_qty = get_position_qty(client, ticker)
    print(f"[Alpaca] Current position: {current_qty} shares")

    # Determine target position from strategy
    if signal == 0 or pos_frac <= 0:
        target_qty = 0
    else:
        target_dollars = equity * pos_frac
        target_qty = int(target_dollars // last_close)

    print(f"Target position: {target_qty} shares")

    # Compare + send orders
    if target_qty == current_qty:
        print("No change required.")
        return

    # Case A: strategy wants FLAT but we have shares -> sell to flatten
    if target_qty == 0 and current_qty > 0:
        print("Signal = FLAT. Flattening existing long position...")
        submit_flatten_order(
            client,
            symbol=ticker,
            qty=current_qty,
            side="sell",
            time_in_force="day",
        )
        return

    # Case B: strategy wants LONG with bracket
    if target_qty > current_qty:
        delta = target_qty - current_qty
        print(f"Entering / increasing LONG by {delta} shares...")

        # Compute ATR-based stop + TP from latest bar
        stop_price, tp_price = compute_bracket_levels(merged)
        # Basic sanity
        if stop_price >= last_close:
            stop_price = last_close * 0.98  # 2% stop as fallback
        if tp_price <= last_close:
            tp_price = last_close * 1.02  # 2% TP as fallback

        submit_bracket_entry(
            client,
            symbol=ticker,
            qty=delta,
            side="buy",
            stop_price=stop_price,
            take_profit_price=tp_price,
            time_in_force="gtc",  # keep stop/TP active across days
        )
        return

    # Case C: strategy wants SMALLER LONG (de-risk) – just trim via market sell
    if target_qty < current_qty:
        delta = current_qty - target_qty
        print(f"Decreasing LONG by {delta} shares (no new bracket legs)...")
        submit_flatten_order(
            client,
            symbol=ticker,
            qty=delta,
            side="sell",
            time_in_force="day",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    run_once(args.ticker, args.start)


if __name__ == "__main__":
    main()