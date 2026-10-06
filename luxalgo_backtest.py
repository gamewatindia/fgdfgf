import argparse
import math
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


# =========================================================
# CONFIG
# =========================================================

API_URL = "https://api.sharkexchange.in/v1/market/klines"

INTERVAL = "5m"
PIVOT_LENGTH = 50

MARGIN_INR = 1000.0
LEVERAGE = 10.0

# Approximate trading costs
TAKER_FEE = 0.00040
GST_ON_FEE = 0.18
SLIPPAGE = 0.00020

# Candle touching support/resistance tolerance
TOUCH_TOLERANCE = 0.0015

REQUEST_LIMIT = 1000

RESULT_DIR = "results"


# =========================================================
# SHARK API
# =========================================================

def shark_request(pair, price_type, end_time=None):
    url = f"{API_URL}?priceType={price_type}"

    payload = {
        "pair": pair,
        "interval": INTERVAL,
        "limit": REQUEST_LIMIT,
    }

    if end_time is not None:
        payload["endTime"] = int(end_time)

    response = requests.post(
        url,
        json=payload,
        timeout=30,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"HTTP ERROR {response.status_code}\n"
            f"{response.text[:1000]}"
        )

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected API response:\n{str(data)[:1000]}"
        )

    return data


def download_history(pair, days, price_type):
    print("=" * 80)
    print("SHARK EXCHANGE HISTORICAL DOWNLOAD")
    print("=" * 80)

    now = datetime.now(timezone.utc)

    requested_end = now
    requested_start = now - timedelta(days=days)

    requested_start_ms = int(requested_start.timestamp() * 1000)

    cursor_end_ms = int(requested_end.timestamp() * 1000)

    all_rows = []

    page = 0

    while True:
        page += 1

        print(
            f"Downloading page {page} "
            f"(through {datetime.fromtimestamp(cursor_end_ms / 1000, tz=timezone.utc)})"
        )

        rows = shark_request(
            pair=pair,
            price_type=price_type,
            end_time=cursor_end_ms,
        )

        if not rows:
            print("No more candles returned.")
            break

        all_rows.extend(rows)

        timestamps = []

        for row in rows:
            try:
                timestamps.append(
                    int(
                        pd.Timestamp(
                            row["startTime"]
                        ).timestamp() * 1000
                    )
                )
            except Exception:
                continue

        if not timestamps:
            break

        oldest_ts = min(timestamps)

        print(
            f"Page {page}: {len(rows)} candles | "
            f"Oldest: {pd.to_datetime(oldest_ts, unit='ms', utc=True)}"
        )

        if oldest_ts <= requested_start_ms:
            break

        next_cursor = oldest_ts - 1

        if next_cursor >= cursor_end_ms:
            print("Pagination stopped: cursor did not move.")
            break

        cursor_end_ms = next_cursor

        # Stay comfortably below API rate limit
        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError("No historical candles downloaded.")

    records = []

    for row in all_rows:
        try:
            ts = pd.Timestamp(row["startTime"])

            records.append(
                {
                    "timestamp": int(ts.timestamp() * 1000),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("Historical data could not be parsed.")

    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Requested period
    df = df[
        (df["timestamp"] >= requested_start_ms)
        & (df["timestamp"] <= int(requested_end.timestamp() * 1000))
    ].copy()

    # Remove currently open candle
    current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    candle_minutes = 5
    candle_ms = candle_minutes * 60 * 1000

    df = df[
        (df["timestamp"] + candle_ms) <= current_ms
    ].copy()

    df = df.reset_index(drop=True)

    print()
    print("DOWNLOAD COMPLETE")
    print(f"Candles : {len(df)}")

    if not df.empty:
        print(
            "From    :",
            pd.to_datetime(df["timestamp"].iloc[0], unit="ms", utc=True),
        )
        print(
            "To      :",
            pd.to_datetime(df["timestamp"].iloc[-1], unit="ms", utc=True),
        )

    expected = int(days * 24 * 60 / 5)

    print(f"Expected: approximately {expected}")

    if len(df) < 200:
        raise RuntimeError(
            f"Not enough candles. Need at least ~200, got {len(df)}"
        )

    return df


# =========================================================
# PIVOT FUNCTIONS
# =========================================================

def is_pivot_high(df, index, length):
    """
    A pivot high at index is confirmed only after
    `length` candles have appeared to the right.
    """

    left = index - length
    right = index + length

    if left < 0 or right >= len(df):
        return False

    value = df["high"].iloc[index]

    left_highs = df["high"].iloc[left:index]
    right_highs = df["high"].iloc[index + 1:right + 1]

    if len(left_highs) != length or len(right_highs) != length:
        return False

    return (
        value >= left_highs.max()
        and value > right_highs.max()
    )


def is_pivot_low(df, index, length):
    """
    A pivot low is confirmed only after `length`
    candles have appeared to the right.
    """

    left = index - length
    right = index + length

    if left < 0 or right >= len(df):
        return False

    value = df["low"].iloc[index]

    left_lows = df["low"].iloc[left:index]
    right_lows = df["low"].iloc[index + 1:right + 1]

    if len(left_lows) != length or len(right_lows) != length:
        return False

    return (
        value <= left_lows.min()
        and value < right_lows.min()
    )


# =========================================================
# LUXALGO-STYLE REVERSAL ENGINE
# =========================================================

def build_dynamic_reversals(df):
    """
    Builds the reversal levels using only information that is
    actually available at each candle.

    IMPORTANT:
    A pivot at P is not available until P + PIVOT_LENGTH.

    This avoids look-ahead bias.

    We maintain:
      - confirmed regular pivot highs
      - confirmed regular pivot lows
      - missed / ghost-style reversal candidates

    The result for each candle contains the first reversal
    level that becomes dynamically available at that candle.
    """

    n = len(df)

    events = [[] for _ in range(n)]

    # -----------------------------------------------------
    # Confirmed regular pivots
    # -----------------------------------------------------

    for confirmation_bar in range(n):

        pivot_bar = confirmation_bar - PIVOT_LENGTH

        if pivot_bar < PIVOT_LENGTH:
            continue

        if pivot_bar + PIVOT_LENGTH >= n:
            continue

        if is_pivot_high(
            df,
            pivot_bar,
            PIVOT_LENGTH,
        ):
            events[confirmation_bar].append(
                {
                    "type": "PIVOT_HIGH",
                    "pivot_index": pivot_bar,
                    "price": float(df["high"].iloc[pivot_bar]),
                }
            )

        if is_pivot_low(
            df,
            pivot_bar,
            PIVOT_LENGTH,
        ):
            events[confirmation_bar].append(
                {
                    "type": "PIVOT_LOW",
                    "pivot_index": pivot_bar,
                    "price": float(df["low"].iloc[pivot_bar]),
                }
            )

    # -----------------------------------------------------
    # LuxAlgo-style state
    # -----------------------------------------------------

    last_pivot_high = None
    last_pivot_low = None

    # These are dynamic ghost/reversal candidates.
    ghost_high = None
    ghost_low = None

    # Each candle receives events that became available
    # at that candle.
    dynamic_events = [[] for _ in range(n)]

    for i in range(n):

        # =================================================
        # New confirmed pivot events
        # =================================================

        for event in events[i]:

            if event["type"] == "PIVOT_HIGH":

                last_pivot_high = event

                dynamic_events[i].append(
                    {
                        "kind": "REGULAR_HIGH",
                        "price": event["price"],
                        "source_index": event["pivot_index"],
                    }
                )

                # A confirmed high can replace an older
                # high-side ghost structure.
                ghost_high = None

            elif event["type"] == "PIVOT_LOW":

                last_pivot_low = event

                dynamic_events[i].append(
                    {
                        "kind": "REGULAR_LOW",
                        "price": event["price"],
                        "source_index": event["pivot_index"],
                    }
                )

                ghost_low = None

        # =================================================
        # Dynamic ghost-style reversal calculation
        # =================================================

        if last_pivot_high is not None:

            pivot_price = last_pivot_high["price"]

            # If price has broken above the last pivot high,
            # LuxAlgo's structure can create a missed/ghost
            # reversal on the opposite side.
            if df["high"].iloc[i] > pivot_price:

                candidate = float(
                    df["low"].iloc[i]
                )

                if ghost_low is None:
                    ghost_low = {
                        "price": candidate,
                        "created_index": i,
                    }

                    dynamic_events[i].append(
                        {
                            "kind": "GHOST_LOW",
                            "price": candidate,
                            "source_index": i,
                        }
                    )

                elif candidate < ghost_low["price"]:

                    ghost_low["price"] = candidate

                    dynamic_events[i].append(
                        {
                            "kind": "GHOST_LOW_SHIFT",
                            "price": candidate,
                            "source_index": i,
                        }
                    )

        if last_pivot_low is not None:

            pivot_price = last_pivot_low["price"]

            if df["low"].iloc[i] < pivot_price:

                candidate = float(
                    df["high"].iloc[i]
                )

                if ghost_high is None:
                    ghost_high = {
                        "price": candidate,
                        "created_index": i,
                    }

                    dynamic_events[i].append(
                        {
                            "kind": "GHOST_HIGH",
                            "price": candidate,
                            "source_index": i,
                        }
                    )

                elif candidate > ghost_high["price"]:

                    ghost_high["price"] = candidate

                    dynamic_events[i].append(
                        {
                            "kind": "GHOST_HIGH_SHIFT",
                            "price": candidate,
                            "source_index": i,
                        }
                    )

    return dynamic_events


# =========================================================
# FIRST DYNAMIC TARGET
# =========================================================

def first_dynamic_target_after_entry(
    df,
    dynamic_events,
    entry_index,
    side,
):
    """
    IMPORTANT:

    We DO NOT search for the nearest old pivot.

    We wait until the first dynamic reversal event appears
    AFTER the trade entry.

    Once the first suitable dynamic reversal appears,
    its price is LOCKED permanently.

    Later shifts are ignored.
    """

    for i in range(entry_index, len(df)):

        candle = df.iloc[i]

        for event in dynamic_events[i]:

            price = float(event["price"])

            if side == "LONG":

                # Target must be above entry.
                if price > float(df["close"].iloc[entry_index]):
                    return {
                        "target": price,
                        "target_index": i,
                        "target_kind": event["kind"],
                    }

            elif side == "SHORT":

                # Target must be below entry.
                if price < float(df["close"].iloc[entry_index]):
                    return {
                        "target": price,
                        "target_index": i,
                        "target_kind": event["kind"],
                    }

    return None


# =========================================================
# P&L
# =========================================================

def calculate_pnl(side, entry, exit_price):
    notional = MARGIN_INR * LEVERAGE

    if side == "LONG":
        gross = notional * (
            (exit_price - entry) / entry
        )
    else:
        gross = notional * (
            (entry - exit_price) / entry
        )

    return gross


def trading_cost(entry, exit_price):
    notional_entry = MARGIN_INR * LEVERAGE
    notional_exit = MARGIN_INR * LEVERAGE

    entry_fee = notional_entry * TAKER_FEE
    exit_fee = notional_exit * TAKER_FEE

    gst = (
        entry_fee + exit_fee
    ) * GST_ON_FEE

    slippage_cost = (
        notional_entry + notional_exit
    ) * SLIPPAGE

    return (
        entry_fee
        + exit_fee
        + gst
        + slippage_cost
    )


# =========================================================
# BACKTEST
# =========================================================

def backtest(df):
    print()
    print("=" * 80)
    print("BUILDING LUXALGO-STYLE DYNAMIC REVERSALS")
    print("=" * 80)

    dynamic_events = build_dynamic_reversals(df)

    print("Dynamic reversal engine ready.")

    trades = []

    i = PIVOT_LENGTH * 2

    while i < len(df) - 2:

        candle = df.iloc[i]

        # =================================================
        # SEARCH FOR CONFIRMATION / ENTRY SETUP
        # =================================================

        side = None
        entry_trigger = None
        sl = None

        # -------------------------------------------------
        # LONG:
        # Look for a dynamic low/reversal that gets touched,
        # followed by bullish confirmation.
        # -------------------------------------------------

        for j in range(max(0, i - 10), i + 1):

            for event in dynamic_events[j]:

                if event["kind"] not in (
                    "REGULAR_LOW",
                    "GHOST_LOW",
                    "GHOST_LOW_SHIFT",
                ):
                    continue

                level = float(event["price"])

                touched = (
                    candle["low"]
                    <= level * (1 + TOUCH_TOLERANCE)
                    and
                    candle["high"]
                    >= level * (1 - TOUCH_TOLERANCE)
                )

                bullish = candle["close"] > candle["open"]

                if touched and bullish:

                    side = "LONG"
                    entry_trigger = float(candle["high"])
                    sl = float(candle["low"])

                    break

            if side:
                break

        # -------------------------------------------------
        # SHORT
        # -------------------------------------------------

        if side is None:

            for j in range(max(0, i - 10), i + 1):

                for event in dynamic_events[j]:

                    if event["kind"] not in (
                        "REGULAR_HIGH",
                        "GHOST_HIGH",
                        "GHOST_HIGH_SHIFT",
                    ):
                        continue

                    level = float(event["price"])

                    touched = (
                        candle["low"]
                        <= level * (1 + TOUCH_TOLERANCE)
                        and
                        candle["high"]
                        >= level * (1 - TOUCH_TOLERANCE)
                    )

                    bearish = candle["close"] < candle["open"]

                    if touched and bearish:

                        side = "SHORT"
                        entry_trigger = float(candle["low"])
                        sl = float(candle["high"])

                        break

                if side:
                    break

        if side is None:
            i += 1
            continue

        # =================================================
        # ENTRY ON NEXT CANDLE BREAK
        # =================================================

        entry_bar = i + 1

        if entry_bar >= len(df):
            break

        next_candle = df.iloc[entry_bar]

        if side == "LONG":

            if next_candle["high"] <= entry_trigger:
                i += 1
                continue

            entry = entry_trigger

        else:

            if next_candle["low"] >= entry_trigger:
                i += 1
                continue

            entry = entry_trigger

        # =================================================
        # FIND FIRST DYNAMIC REVERSAL AFTER ENTRY
        # =================================================

        target_info = first_dynamic_target_after_entry(
            df=df,
            dynamic_events=dynamic_events,
            entry_index=entry_bar,
            side=side,
        )

        if target_info is None:
            i += 1
            continue

        target = target_info["target"]

        target_index = target_info["target_index"]

        target_kind = target_info["target_kind"]

        # =================================================
        # TARGET IS NOW LOCKED
        # =================================================

        exit_price = None
        exit_index = None
        exit_reason = None

        for k in range(entry_bar, len(df)):

            bar = df.iloc[k]

            # -------------------------------------------------
            # SL / TARGET collision handling
            # -------------------------------------------------

            if side == "LONG":

                hit_sl = bar["low"] <= sl
                hit_target = bar["high"] >= target

                if hit_sl and hit_target:

                    # Conservative: assume SL first.
                    exit_price = sl
                    exit_index = k
                    exit_reason = "SL_AND_TARGET_SAME_CANDLE"

                    break

                if hit_sl:

                    exit_price = sl
                    exit_index = k
                    exit_reason = "SL"

                    break

                if hit_target:

                    exit_price = target
                    exit_index = k
                    exit_reason = "DYNAMIC_REVERSAL_TARGET"

                    break

            else:

                hit_sl = bar["high"] >= sl
                hit_target = bar["low"] <= target

                if hit_sl and hit_target:

                    exit_price = sl
                    exit_index = k
                    exit_reason = "SL_AND_TARGET_SAME_CANDLE"

                    break

                if hit_sl:

                    exit_price = sl
                    exit_index = k
                    exit_reason = "SL"

                    break

                if hit_target:

                    exit_price = target
                    exit_index = k
                    exit_reason = "DYNAMIC_REVERSAL_TARGET"

                    break

        if exit_price is None:
            break

        gross_pnl = calculate_pnl(
            side,
            entry,
            exit_price,
        )

        costs = trading_cost(
            entry,
            exit_price,
        )

        net_pnl = gross_pnl - costs

        risk_price = (
            abs(entry - sl)
        )

        if risk_price > 0:

            if side == "LONG":
                reward_price = exit_price - entry
            else:
                reward_price = entry - exit_price

            rr = reward_price / risk_price

        else:
            rr = 0.0

        trades.append(
            {
                "entry_time": pd.to_datetime(
                    df["timestamp"].iloc[entry_bar],
                    unit="ms",
                    utc=True,
                ),
                "exit_time": pd.to_datetime(
                    df["timestamp"].iloc[exit_index],
                    unit="ms",
                    utc=True,
                ),
                "side": side,
                "entry": entry,
                "sl": sl,
                "target": target,
                "target_appeared_time": pd.to_datetime(
                    df["timestamp"].iloc[target_index],
                    unit="ms",
                    utc=True,
                ),
                "target_kind": target_kind,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "RR": rr,
                "gross_pnl": gross_pnl,
                "fees_slippage": costs,
                "net_pnl": net_pnl,
            }
        )

        # No overlapping trades.
        i = exit_index + 1

    return trades


# =========================================================
# SUMMARY
# =========================================================

def save_results(trades):

    os.makedirs(RESULT_DIR, exist_ok=True)

    trades_file = os.path.join(
        RESULT_DIR,
        "trades_luxalgo_dynamic_first_reversal.csv",
    )

    summary_file = os.path.join(
        RESULT_DIR,
        "summary.csv",
    )

    if not trades:

        print("No trades generated.")

        pd.DataFrame(
            [
                {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0,
                    "gross_pnl": 0,
                    "fees": 0,
                    "net_pnl": 0,
                    "profit_factor": 0,
                }
            ]
        ).to_csv(
            summary_file,
            index=False,
        )

        return

    df = pd.DataFrame(trades)

    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] < 0]

    gross_profit = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else math.inf
    )

    summary = {
        "trades": len(df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (
            len(wins) / len(df) * 100
        ),
        "gross_pnl": df["gross_pnl"].sum(),
        "fees": df["fees_slippage"].sum(),
        "net_pnl": df["net_pnl"].sum(),
        "profit_factor": profit_factor,
        "average_pnl": df["net_pnl"].mean(),
        "average_RR": df["RR"].mean(),
        "max_win": df["net_pnl"].max(),
        "max_loss": df["net_pnl"].min(),
    }

    df.to_csv(
        trades_file,
        index=False,
    )

    pd.DataFrame([summary]).to_csv(
        summary_file,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(f"Trades        : {summary['trades']}")
    print(f"Wins          : {summary['wins']}")
    print(f"Losses        : {summary['losses']}")
    print(f"Win rate      : {summary['win_rate']:.2f}%")
    print(f"Gross P&L     : ₹{summary['gross_pnl']:.2f}")
    print(f"Fees + slip   : ₹{summary['fees']:.2f}")
    print(f"NET P&L       : ₹{summary['net_pnl']:.2f}")
    print(f"Profit Factor : {summary['profit_factor']:.3f}")
    print(f"Average RR    : {summary['average_RR']:.2f}")

    print()
    print(f"Saved: {trades_file}")
    print(f"Saved: {summary_file}")


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair",
        default="BRUSDT",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--price-type",
        default="LAST_PRICE",
        choices=[
            "LAST_PRICE",
            "MARK_PRICE",
        ],
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LUXALGO FIRST DYNAMIC REVERSAL TARGET BACKTEST")
    print("=" * 80)

    print(f"Pair       : {args.pair}")
    print(f"Timeframe  : {INTERVAL}")
    print(f"Pivot      : {PIVOT_LENGTH}")
    print(f"Price type : {args.price_type}")
    print(f"Days       : {args.days}")
    print(f"Margin     : ₹{MARGIN_INR}")
    print(f"Leverage   : {LEVERAGE}x")

    df = download_history(
        pair=args.pair,
        days=args.days,
        price_type=args.price_type,
    )

    trades = backtest(df)

    save_results(trades)


if __name__ == "__main__":
    main()
