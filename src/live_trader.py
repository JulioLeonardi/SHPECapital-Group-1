import argparse
import datetime as dt

from config import StrategyConfig
from data import fetch_price, fetch_refs
from strategy import build_strategy_signals
from broker_alpaca import (get_client, get_account_equity, get_position_qty, submit_market_order)


def run_once(ticker: str, start: str = "2015-01-01"):
    """
    compute latest signal/position size and rebalance Alpaca 
    paper account to the target position in `ticker`.
    """
    end = dt.date.today().strftime("%Y-%m-%d")

    # Get price & reference series
    tdf = fetch_price(ticker, start, end)  # target series
    if tdf.empty:
        raise RuntimeError(f"No data for {ticker} between {start} and {end}")

    market_df, vix_df, sector_df = fetch_refs(ticker, start, end)

    # Build factor scores, combo score, signal, and position size
    cfg = StrategyConfig()
    sig_df = build_strategy_signals(
        tdf=tdf,
        market_df=market_df,
        sector_df=sector_df,
        vix_df=vix_df,
        cfg=cfg,
    )

    full = tdf.join(sig_df, how="inner")
    last = full.iloc[-1]

    close = float(last["Close"])
    signal = int(last["Signal"])
    pos_size = float(last["PosSize"])

    print(f"Latest date: {last.name.date()}")
    print(f"Close={close:.2f}, Score={last['Score']:.3f}, Signal={signal}, PosSize={pos_size:.3f}")

    # Interpret PosSize as "fraction of equity" (0..max_leverage)
    if signal == 0 or pos_size <= 0:
        target_weight = 0.0
    else:
        target_weight = pos_size  # can rescale or cap if you like

    # Connect to Alpaca and compute target shares
    client = get_client()
    equity = get_account_equity(client)
    print(f"Account equity: {equity:.2f} USD")

    target_dollars = equity * target_weight
    target_qty = int(target_dollars // close) if close > 0 else 0

    current_qty = get_position_qty(client, ticker)
    delta = target_qty - current_qty

    print(f"Current qty: {current_qty}, Target qty: {target_qty}, Delta: {delta}")

    # 4) Fire orders to move toward target
    if delta > 0:
        submit_market_order(client, ticker, "buy", delta)
    elif delta < 0:
        submit_market_order(client, ticker, "sell", -delta)
    else:
        print("No change in position needed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    run_once(args.ticker, start=args.start)


if __name__ == "__main__":
    main()
