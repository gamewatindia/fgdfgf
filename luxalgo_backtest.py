import argparse
import time
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


# =========================================================
# CONFIG
# =========================================================

API_URL = "https://api.sharkexchange.in/v1/market/klines"

PIVOT_LENGTH = 50

MARGIN = 1000.0
LEVERAGE = 10.0

TOUCH_TOLERANCE = 0.0015

FEE_RATE = 0.00040
GST = 0.18
SLIPPAGE = 0.00020

LIMIT = 1000


# =========================================================
# DOWNLOAD SHARK KLINES
# =========================================================

def download_klines(pair, days, price_type):

    end_time = datetime.now(timezone.utc)

    start_time = (
        end_time -
        timedelta(days=days)
    )

    start_ms = int(
        start_time.timestamp() * 1000
    )

    end_ms = int(
        end_time.timestamp() * 1000
    )

    interval_ms = 5 * 60 * 1000

    all_data = []

    cursor = start_ms

    session = requests.Session()

    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Content-Type": "application/json"
    })

    print("=" * 60)
    print("SHARK EXCHANGE DATA")
    print("=" * 60)

    print(f"Pair       : {pair}")
    print("Interval   : 5m")
    print(f"Days       : {days}")
    print(f"Price type : {price_type}")
    print()

    page = 0

    while cursor < end_ms:

        page += 1

        params = {
            "pair": pair.upper(),
            "interval": "5m",
            "priceType": price_type,
            "limit": LIMIT
        }

        # Shark SDK documents pair/interval/priceType/limit
        # for public market klines.

        try:

            response = session.get(
                API_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()

        except Exception as e:

            print(
                f"Page {page} error: {e}"
            )

            time.sleep(3)

            page -= 1

            continue

        if isinstance(result, dict):

            data = result.get(
                "data",
                []
            )

        else:

            data = result

        if not data:

            print(
                "No more candles returned."
            )

            break

        # -------------------------------------------------
        # Parse candles
        # -------------------------------------------------

        parsed = []

        for row in data:

            try:

                if isinstance(row, list):

                    if len(row) < 6:
                        continue

                    ts = int(row[0])

                    o = float(row[1])
                    h = float(row[2])
                    l = float(row[3])
                    c = float(row[4])

                    v = float(row[5])

                    end_ts = (
                        int(row[6])
                        if len(row) > 6
                        else ts + interval_ms - 1
                    )

                elif isinstance(row, dict):

                    ts = int(
                        row.get(
                            "startTime",
                            row.get(
                                "timestamp"
                            )
                        )
                    )

                    o = float(
                        row["open"]
                    )

                    h = float(
                        row["high"]
                    )

                    l = float(
                        row["low"]
                    )

                    c = float(
                        row["close"]
                    )

                    v = float(
                        row.get(
                            "volume",
                            0
                        )
                    )

                    end_ts = int(
                        row.get(
                            "endTime",
                            ts +
                            interval_ms -
                            1
                        )
                    )

                else:

                    continue

                parsed.append({

                    "timestamp": pd.to_datetime(
                        ts,
                        unit="ms",
                        utc=True
                    ),

                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,

                    "endTime": end_ts,

                    "_ts": ts
                })

            except Exception:

                continue

        if not parsed:

            print(
                "Unable to parse API response."
            )

            break

        all_data.extend(
            parsed
        )

        newest = max(
            x["_ts"]
            for x in parsed
        )

        oldest = min(
            x["_ts"]
            for x in parsed
        )

        print(
            f"Page {page}: "
            f"{len(parsed)} candles | "
            f"{pd.to_datetime(newest, unit='ms', utc=True)}"
        )

        # -------------------------------------------------
        # Pagination
        # -------------------------------------------------

        next_cursor = (
            newest +
            interval_ms
        )

        if next_cursor <= cursor:

            print(
                "Pagination stopped."
            )

            break

        cursor = next_cursor

        # If API gives only a small page but has not
        # reached requested end, continue anyway.
        if newest >= end_ms:

            break

        time.sleep(0.20)

    if not all_data:

        raise RuntimeError(
            "No Shark candle data received."
        )

    df = pd.DataFrame(
        all_data
    )

    df.drop_duplicates(
        subset=["_ts"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "_ts",
        inplace=True
    )

    # Requested period
    df = df[
        (df["_ts"] >= start_ms) &
        (df["_ts"] <= end_ms)
    ].copy()

    # Remove current candle
    current_ms = int(
        time.time() * 1000
    )

    df = df[
        df["endTime"] < current_ms
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    if len(df) < 200:

        raise RuntimeError(
            f"Only {len(df)} closed candles found."
        )

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(
        f"Candles : {len(df):,}"
    )

    print(
        f"From    : {df['timestamp'].iloc[0]}"
    )

    print(
        f"To      : {df['timestamp'].iloc[-1]}"
    )

    return df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]


# =========================================================
# PIVOT
# =========================================================

def is_pivot_high(df, i):

    if (
        i < PIVOT_LENGTH
        or
        i + PIVOT_LENGTH >= len(df)
    ):
        return False

    value = float(
        df.iloc[i]["high"]
    )

    left = df.iloc[
        i - PIVOT_LENGTH:i
    ]["high"]

    right = df.iloc[
        i + 1:
        i + PIVOT_LENGTH + 1
    ]["high"]

    return (
        value >= left.max()
        and
        value >= right.max()
    )


def is_pivot_low(df, i):

    if (
        i < PIVOT_LENGTH
        or
        i + PIVOT_LENGTH >= len(df)
    ):
        return False

    value = float(
        df.iloc[i]["low"]
    )

    left = df.iloc[
        i - PIVOT_LENGTH:i
    ]["low"]

    right = df.iloc[
        i + 1:
        i + PIVOT_LENGTH + 1
    ]["low"]

    return (
        value <= left.min()
        and
        value <= right.min()
    )


# =========================================================
# FEE
# =========================================================

def calculate_fee(notional):

    return (
        notional *
        FEE_RATE *
        (1 + GST)
    )


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(df, symbol):

    resistance = []
    support = []

    pending = None
    position = None

    trades = []

    last_pivot = None

    running_high = None
    running_low = None

    for i in range(
        PIVOT_LENGTH * 2,
        len(df)
    ):

        candle = df.iloc[i]

        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])

        # =================================================
        # CONFIRMED PIVOT
        # =================================================

        p = i - PIVOT_LENGTH

        if is_pivot_high(df, p):

            pivot_high = float(
                df.iloc[p]["high"]
            )

            # Missed reversal support
            if (
                last_pivot == "HIGH"
                and
                running_low is not None
            ):

                support.append(
                    running_low
                )

            resistance.append(
                pivot_high
            )

            last_pivot = "HIGH"

            running_high = pivot_high
            running_low = float(
                df.iloc[p]["low"]
            )

        if is_pivot_low(df, p):

            pivot_low = float(
                df.iloc[p]["low"]
            )

            # Missed reversal resistance
            if (
                last_pivot == "LOW"
                and
                running_high is not None
            ):

                resistance.append(
                    running_high
                )

            support.append(
                pivot_low
            )

            last_pivot = "LOW"

            running_low = pivot_low
            running_high = float(
                df.iloc[p]["high"]
            )

        # Update extremes
        if (
            running_high is None
            or
            h > running_high
        ):
            running_high = h

        if (
            running_low is None
            or
            l < running_low
        ):
            running_low = l

        # =================================================
        # MANAGE OPEN POSITION
        # =================================================

        if position is not None:

            side = position["side"]

            if side == "LONG":

                sl_hit = (
                    l <= position["stop"]
                )

                target_hit = (
                    h >= position["target"]
                )

            else:

                sl_hit = (
                    h >= position["stop"]
                )

                target_hit = (
                    l <= position["target"]
                )

            exit_reason = None
            exit_raw = None

            if sl_hit:

                exit_reason = "SL"
                exit_raw = position["stop"]

            elif target_hit:

                exit_reason = "TARGET"
                exit_raw = position["target"]

            if exit_reason:

                if side == "LONG":

                    exit_actual = (
                        exit_raw *
                        (1 - SLIPPAGE)
                    )

                    gross = (
                        exit_actual -
                        position["entry"]
                    ) * position["qty"]

                else:

                    exit_actual = (
                        exit_raw *
                        (1 + SLIPPAGE)
                    )

                    gross = (
                        position["entry"] -
                        exit_actual
                    ) * position["qty"]

                exit_fee = calculate_fee(
                    exit_actual *
                    position["qty"]
                )

                total_fee = (
                    position["entry_fee"]
                    +
                    exit_fee
                )

                net = (
                    gross -
                    total_fee
                )

                trades.append({

                    "symbol": symbol,

                    "side": side,

                    "signal_time":
                        position["signal_time"],

                    "entry_time":
                        position["entry_time"],

                    "exit_time":
                        candle[
                            "timestamp"
                        ].isoformat(),

                    "entry":
                        position["entry"],

                    "stop":
                        position["stop"],

                    "target":
                        position["target"],

                    "exit":
                        exit_actual,

                    "qty":
                        position["qty"],

                    "margin":
                        MARGIN,

                    "leverage":
                        LEVERAGE,

                    "planned_rr":
                        position["planned_rr"],

                    "gross_pnl":
                        gross,

                    "fees":
                        total_fee,

                    "net_pnl":
                        net,

                    "exit_reason":
                        exit_reason,

                    "target_source":
                        position[
                            "target_source"
                        ]
                })

                position = None

                continue

        # =================================================
        # PENDING ENTRY
        # =================================================

        if (
            pending is not None
            and
            position is None
            and
            i > pending["signal_bar"]
        ):

            side = pending["side"]

            if side == "LONG":

                if h > pending["trigger"]:

                    entry = (
                        pending["trigger"] *
                        (1 + SLIPPAGE)
                    )

                    stop = pending["stop"]
                    target = pending["target"]

                    risk = (
                        entry -
                        stop
                    )

                    if (
                        risk > 0
                        and
                        target > entry
                    ):

                        rr = (
                            target -
                            entry
                        ) / risk

                        qty = (
                            MARGIN *
                            LEVERAGE /
                            entry
                        )

                        position = {

                            "side": "LONG",

                            "entry": entry,

                            "stop": stop,

                            "target": target,

                            "qty": qty,

                            "entry_fee":
                                calculate_fee(
                                    entry * qty
                                ),

                            "signal_time":
                                pending[
                                    "signal_time"
                                ],

                            "entry_time":
                                candle[
                                    "timestamp"
                                ].isoformat(),

                            "planned_rr": rr,

                            "target_source":
                                pending[
                                    "target_source"
                                ]
                        }

                    pending = None

            else:

                if l < pending["trigger"]:

                    entry = (
                        pending["trigger"] *
                        (1 - SLIPPAGE)
                    )

                    stop = pending["stop"]
                    target = pending["target"]

                    risk = (
                        stop -
                        entry
                    )

                    if (
                        risk > 0
                        and
                        target < entry
                    ):

                        rr = (
                            entry -
                            target
                        ) / risk

                        qty = (
                            MARGIN *
                            LEVERAGE /
                            entry
                        )

                        position = {

                            "side": "SHORT",

                            "entry": entry,

                            "stop": stop,

                            "target": target,

                            "qty": qty,

                            "entry_fee":
                                calculate_fee(
                                    entry * qty
                                ),

                            "signal_time":
                                pending[
                                    "signal_time"
                                ],

                            "entry_time":
                                candle[
                                    "timestamp"
                                ].isoformat(),

                            "planned_rr": rr,

                            "target_source":
                                pending[
                                    "target_source"
                                ]
                        }

                    pending = None

        # =================================================
        # NEW SIGNAL
        # =================================================

        if (
            pending is None
            and
            position is None
        ):

            # LONG: support touch + bullish candle

            supports = [
                x for x in support
                if (
                    abs(l - x) / x
                    <= TOUCH_TOLERANCE
                    and
                    c > x
                )
            ]

            if (
                c > o
                and
                supports
            ):

                target_levels = [
                    x for x in resistance
                    if x > c
                ]

                if target_levels:

                    target = min(
                        target_levels
                    )

                    pending = {

                        "side": "LONG",

                        "signal_bar": i,

                        "signal_time":
                            candle[
                                "timestamp"
                            ].isoformat(),

                        "trigger": h,

                        "stop": l,

                        "target": target,

                        "target_source":
                            "nearest_reversal_resistance"
                    }

            # SHORT: resistance touch + bearish

            else:

                resistances = [
                    x for x in resistance
                    if (
                        abs(h - x) / x
                        <= TOUCH_TOLERANCE
                        and
                        c < x
                    )
                ]

                if (
                    c < o
                    and
                    resistances
                ):

                    target_levels = [
                        x for x in support
                        if x < c
                    ]

                    if target_levels:

                        target = max(
                            target_levels
                        )

                        pending = {

                            "side": "SHORT",

                            "signal_bar": i,

                            "signal_time":
                                candle[
                                    "timestamp"
                                ].isoformat(),

                            "trigger": l,

                            "stop": h,

                            "target": target,

                            "target_source":
                                "nearest_reversal_support"
                        }

    return pd.DataFrame(trades)


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair",
        default="BRUSDT"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30
    )

    parser.add_argument(
        "--price-type",
        default="LAST_PRICE"
    )

    args = parser.parse_args()

    df = download_klines(
        args.pair,
        args.days,
        args.price_type
    )

    print()
    print("=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)

    trades = run_backtest(
        df,
        args.pair
    )

    out = Path("results")

    out.mkdir(
        exist_ok=True
    )

    trades_file = (
        out /
        "trades_luxalgo_reversal.csv"
    )

    summary_file = (
        out /
        "summary.csv"
    )

    trades.to_csv(
        trades_file,
        index=False
    )

    if trades.empty:

        summary = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_pnl": 0,
            "fees": 0,
            "profit_factor": 0
        }

    else:

        wins = trades[
            trades["net_pnl"] > 0
        ]

        losses = trades[
            trades["net_pnl"] <= 0
        ]

        gross_profit = float(
            wins["net_pnl"].sum()
        )

        gross_loss = abs(
            float(
                losses["net_pnl"].sum()
            )
        )

        summary = {

            "trades":
                len(trades),

            "wins":
                len(wins),

            "losses":
                len(losses),

            "win_rate":
                round(
                    len(wins) /
                    len(trades) *
                    100,
                    2
                ),

            "net_pnl":
                round(
                    float(
                        trades[
                            "net_pnl"
                        ].sum()
                    ),
                    2
                ),

            "fees":
                round(
                    float(
                        trades[
                            "fees"
                        ].sum()
                    ),
                    2
                ),

            "profit_factor":
                round(
                    gross_profit /
                    gross_loss,
                    3
                )
                if gross_loss > 0
                else "INF"
        }

    pd.DataFrame([
        summary
    ]).to_csv(
        summary_file,
        index=False
    )

    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)

    for key, value in summary.items():

        print(
            f"{key:16}: {value}"
        )

    print()
    print(
        f"Trades  : {trades_file}"
    )

    print(
        f"Summary : {summary_file}"
    )


if __name__ == "__main__":
    main()
