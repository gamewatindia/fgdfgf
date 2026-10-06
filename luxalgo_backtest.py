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

MARGIN_INR = 1000.0
LEVERAGE = 10.0

TOUCH_TOLERANCE = 0.0015

TAKER_FEE = 0.00040
GST_ON_FEE = 0.18
SLIPPAGE = 0.00020

LIMIT = 1000

INTERVAL_MS = 5 * 60 * 1000


# =========================================================
# SHARK API - BACKWARD PAGINATION
# =========================================================

def download_klines(pair, days, price_type):

    now = datetime.now(timezone.utc)

    requested_start = (
        now - timedelta(days=days)
    )

    requested_start_ms = int(
        requested_start.timestamp() * 1000
    )

    requested_end_ms = int(
        now.timestamp() * 1000
    )

    # We start from the present and move BACKWARD.
    cursor_end = requested_end_ms

    all_rows = []

    page = 0

    session = requests.Session()

    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    })

    print()
    print("=" * 60)
    print("SHARK EXCHANGE DATA")
    print("=" * 60)

    print(f"Pair       : {pair}")
    print("Interval   : 5m")
    print(f"Days       : {days}")
    print(f"Price type : {price_type}")
    print()

    while cursor_end > requested_start_ms:

        page += 1

        # -------------------------------------------------
        # IMPORTANT:
        #
        # Shark returns the latest 1000 candles before
        # endTime.
        #
        # Therefore we paginate BACKWARD.
        # -------------------------------------------------

        url = (
            f"{API_URL}"
            f"?priceType={price_type}"
        )

        payload = {
            "pair": pair.upper(),
            "interval": "5m",
            "endTime": cursor_end,
            "limit": LIMIT
        }

        try:

            response = session.post(
                url,
                json=payload,
                timeout=30
            )

            # 200 and 201 are both accepted because
            # the Shark endpoint has returned 201 in
            # the user's GitHub run.

            if response.status_code not in (200, 201):

                print()
                print(
                    f"HTTP ERROR {response.status_code}"
                )

                print(
                    response.text[:1000]
                )

                response.raise_for_status()

            result = response.json()

        except Exception as e:

            print(
                f"Page {page} ERROR: {e}"
            )

            raise

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        if isinstance(result, dict):

            data = result.get(
                "data",
                []
            )

        elif isinstance(result, list):

            data = result

        else:

            data = []

        if not data:

            print(
                f"Page {page}: "
                "No more candles."
            )

            break

        parsed = []

        for row in data:

            try:

                if isinstance(row, dict):

                    ts = int(
                        row.get(
                            "startTime",
                            row.get(
                                "timestamp"
                            )
                        )
                    )

                    open_price = float(
                        row["open"]
                    )

                    high_price = float(
                        row["high"]
                    )

                    low_price = float(
                        row["low"]
                    )

                    close_price = float(
                        row["close"]
                    )

                    volume = float(
                        row.get(
                            "volume",
                            0
                        )
                    )

                    end_time = int(
                        row.get(
                            "endTime",
                            ts +
                            INTERVAL_MS -
                            1
                        )
                    )

                elif isinstance(
                    row,
                    (list, tuple)
                ):

                    if len(row) < 5:
                        continue

                    ts = int(row[0])

                    open_price = float(row[1])
                    high_price = float(row[2])
                    low_price = float(row[3])
                    close_price = float(row[4])

                    volume = (
                        float(row[5])
                        if len(row) > 5
                        else 0.0
                    )

                    end_time = (
                        int(row[6])
                        if len(row) > 6
                        else
                        ts +
                        INTERVAL_MS -
                        1
                    )

                else:

                    continue

                parsed.append({

                    "timestamp":
                        pd.to_datetime(
                            ts,
                            unit="ms",
                            utc=True
                        ),

                    "open":
                        open_price,

                    "high":
                        high_price,

                    "low":
                        low_price,

                    "close":
                        close_price,

                    "volume":
                        volume,

                    "endTime":
                        end_time,

                    "_ts":
                        ts
                })

            except Exception:

                continue

        if not parsed:

            raise RuntimeError(
                "Shark returned data, "
                "but candles could not be parsed."
            )

        # Add page
        all_rows.extend(parsed)

        # -------------------------------------------------
        # Find OLDEST candle in this page
        # -------------------------------------------------

        oldest_ts = min(
            x["_ts"]
            for x in parsed
        )

        newest_ts = max(
            x["_ts"]
            for x in parsed
        )

        print(
            f"Page {page}: "
            f"{len(parsed)} candles | "
            f"{pd.to_datetime(oldest_ts, unit='ms', utc=True)}"
            f" -> "
            f"{pd.to_datetime(newest_ts, unit='ms', utc=True)}"
        )

        # -------------------------------------------------
        # BACKWARD PAGINATION
        # -------------------------------------------------

        next_cursor = (
            oldest_ts - 1
        )

        if next_cursor >= cursor_end:

            raise RuntimeError(
                "Backward pagination did not move."
            )

        cursor_end = next_cursor

        # -------------------------------------------------
        # Stop once we have crossed requested start
        # -------------------------------------------------

        if oldest_ts <= requested_start_ms:

            break

        # Shark public endpoint has rate limits.
        time.sleep(0.20)

    # =====================================================
    # NO DATA
    # =====================================================

    if not all_rows:

        raise RuntimeError(
            "No candle data received from Shark Exchange."
        )

    # =====================================================
    # DATAFRAME
    # =====================================================

    df = pd.DataFrame(
        all_rows
    )

    # Remove duplicates
    df.drop_duplicates(
        subset=["_ts"],
        keep="last",
        inplace=True
    )

    # Sort oldest -> newest
    df.sort_values(
        "_ts",
        inplace=True
    )

    # Keep requested period
    df = df[
        (df["_ts"] >= requested_start_ms) &
        (df["_ts"] <= requested_end_ms)
    ].copy()

    # =====================================================
    # REMOVE CURRENT OPEN CANDLE
    # =====================================================

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
            f"Only {len(df)} closed candles "
            f"received. Expected much more."
        )

    # =====================================================
    # DOWNLOAD SUMMARY
    # =====================================================

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(
        f"Candles : {len(df):,}"
    )

    print(
        f"From    : "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"To      : "
        f"{df['timestamp'].iloc[-1]}"
    )

    expected = (
        days *
        24 *
        12
    )

    print(
        f"Expected: ~{expected:,}"
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
    ].copy()


# =========================================================
# PIVOT HIGH
# =========================================================

def is_pivot_high(df, index):

    if index < PIVOT_LENGTH:
        return False

    if (
        index +
        PIVOT_LENGTH
        >= len(df)
    ):
        return False

    value = float(
        df.iloc[index]["high"]
    )

    left = df.iloc[
        index - PIVOT_LENGTH:index
    ]["high"]

    right = df.iloc[
        index + 1:
        index + PIVOT_LENGTH + 1
    ]["high"]

    return (
        value >= left.max()
        and
        value >= right.max()
    )


# =========================================================
# PIVOT LOW
# =========================================================

def is_pivot_low(df, index):

    if index < PIVOT_LENGTH:
        return False

    if (
        index +
        PIVOT_LENGTH
        >= len(df)
    ):
        return False

    value = float(
        df.iloc[index]["low"]
    )

    left = df.iloc[
        index - PIVOT_LENGTH:index
    ]["low"]

    right = df.iloc[
        index + 1:
        index + PIVOT_LENGTH + 1
    ]["low"]

    return (
        value <= left.min()
        and
        value <= right.min()
    )


# =========================================================
# FEES
# =========================================================

def calculate_fee(notional):

    trading_fee = (
        notional *
        TAKER_FEE
    )

    return (
        trading_fee *
        (1 + GST_ON_FEE)
    )


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(df, symbol):

    resistance_levels = []
    support_levels = []

    pending = None
    position = None

    trades = []

    last_pivot_type = None

    running_high = None
    running_low = None

    # =====================================================
    # LOOP
    # =====================================================

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
        # CONFIRM PIVOT
        # =================================================

        pivot_index = (
            i -
            PIVOT_LENGTH
        )

        # -------------------------------------------------
        # PIVOT HIGH
        # -------------------------------------------------

        if is_pivot_high(
            df,
            pivot_index
        ):

            pivot_price = float(
                df.iloc[
                    pivot_index
                ]["high"]
            )

            # Missed reversal support
            if (
                last_pivot_type == "HIGH"
                and
                running_low is not None
            ):

                support_levels.append(
                    running_low
                )

            resistance_levels.append(
                pivot_price
            )

            last_pivot_type = "HIGH"

            running_high = pivot_price

            running_low = float(
                df.iloc[
                    pivot_index
                ]["low"]
            )

        # -------------------------------------------------
        # PIVOT LOW
        # -------------------------------------------------

        if is_pivot_low(
            df,
            pivot_index
        ):

            pivot_price = float(
                df.iloc[
                    pivot_index
                ]["low"]
            )

            # Missed reversal resistance
            if (
                last_pivot_type == "LOW"
                and
                running_high is not None
            ):

                resistance_levels.append(
                    running_high
                )

            support_levels.append(
                pivot_price
            )

            last_pivot_type = "LOW"

            running_low = pivot_price

            running_high = float(
                df.iloc[
                    pivot_index
                ]["high"]
            )

        # =================================================
        # UPDATE EXTREMES
        # =================================================

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

                hit_sl = (
                    l <= position["stop"]
                )

                hit_target = (
                    h >= position["target"]
                )

            else:

                hit_sl = (
                    h >= position["stop"]
                )

                hit_target = (
                    l <= position["target"]
                )

            exit_reason = None
            raw_exit = None

            # SL has priority when both
            # are touched inside one candle.

            if hit_sl:

                exit_reason = "SL"

                raw_exit = (
                    position["stop"]
                )

            elif hit_target:

                exit_reason = "TARGET"

                raw_exit = (
                    position["target"]
                )

            if exit_reason:

                # -------------------------------------------------
                # Exit price with slippage
                # -------------------------------------------------

                if side == "LONG":

                    actual_exit = (
                        raw_exit *
                        (1 - SLIPPAGE)
                    )

                    gross_pnl = (
                        actual_exit -
                        position["entry"]
                    ) * position["qty"]

                else:

                    actual_exit = (
                        raw_exit *
                        (1 + SLIPPAGE)
                    )

                    gross_pnl = (
                        position["entry"] -
                        actual_exit
                    ) * position["qty"]

                # -------------------------------------------------
                # Fees
                # -------------------------------------------------

                exit_fee = calculate_fee(
                    actual_exit *
                    position["qty"]
                )

                total_fee = (
                    position["entry_fee"]
                    +
                    exit_fee
                )

                net_pnl = (
                    gross_pnl -
                    total_fee
                )

                # -------------------------------------------------
                # RR
                # -------------------------------------------------

                risk = abs(
                    position["entry"] -
                    position["stop"]
                )

                if side == "LONG":

                    reward = (
                        position["target"] -
                        position["entry"]
                    )

                else:

                    reward = (
                        position["entry"] -
                        position["target"]
                    )

                planned_rr = (
                    reward / risk
                    if risk > 0
                    else 0
                )

                trades.append({

                    "symbol":
                        symbol,

                    "side":
                        side,

                    "signal_time":
                        position[
                            "signal_time"
                        ],

                    "entry_time":
                        position[
                            "entry_time"
                        ],

                    "exit_time":
                        candle[
                            "timestamp"
                        ].isoformat(),

                    "entry":
                        position[
                            "entry"
                        ],

                    "stop":
                        position[
                            "stop"
                        ],

                    "target":
                        position[
                            "target"
                        ],

                    "exit":
                        actual_exit,

                    "planned_rr":
                        planned_rr,

                    "qty":
                        position[
                            "qty"
                        ],

                    "margin":
                        MARGIN_INR,

                    "leverage":
                        LEVERAGE,

                    "gross_pnl_inr":
                        gross_pnl,

                    "fees_inr":
                        total_fee,

                    "net_pnl_inr":
                        net_pnl,

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

            # -------------------------------------------------
            # LONG
            # -------------------------------------------------

            if side == "LONG":

                if h > pending["trigger"]:

                    entry = (
                        pending["trigger"]
                        *
                        (1 + SLIPPAGE)
                    )

                    stop = (
                        pending["stop"]
                    )

                    target = (
                        pending["target"]
                    )

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
                            MARGIN_INR *
                            LEVERAGE /
                            entry
                        )

                        entry_fee = calculate_fee(
                            entry * qty
                        )

                        position = {

                            "side":
                                "LONG",

                            "entry":
                                entry,

                            "stop":
                                stop,

                            "target":
                                target,

                            "qty":
                                qty,

                            "entry_fee":
                                entry_fee,

                            "signal_time":
                                pending[
                                    "signal_time"
                                ],

                            "entry_time":
                                candle[
                                    "timestamp"
                                ].isoformat(),

                            "planned_rr":
                                rr,

                            "target_source":
                                pending[
                                    "target_source"
                                ]
                        }

                    pending = None

            # -------------------------------------------------
            # SHORT
            # -------------------------------------------------

            else:

                if l < pending["trigger"]:

                    entry = (
                        pending["trigger"]
                        *
                        (1 - SLIPPAGE)
                    )

                    stop = (
                        pending["stop"]
                    )

                    target = (
                        pending["target"]
                    )

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
                            MARGIN_INR *
                            LEVERAGE /
                            entry
                        )

                        entry_fee = calculate_fee(
                            entry * qty
                        )

                        position = {

                            "side":
                                "SHORT",

                            "entry":
                                entry,

                            "stop":
                                stop,

                            "target":
                                target,

                            "qty":
                                qty,

                            "entry_fee":
                                entry_fee,

                            "signal_time":
                                pending[
                                    "signal_time"
                                ],

                            "entry_time":
                                candle[
                                    "timestamp"
                                ].isoformat(),

                            "planned_rr":
                                rr,

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

            # -------------------------------------------------
            # LONG
            # -------------------------------------------------

            support_hits = [

                level

                for level
                in support_levels

                if (
                    level > 0
                    and
                    abs(
                        l - level
                    ) / level
                    <= TOUCH_TOLERANCE
                    and
                    c > level
                )
            ]

            if (
                c > o
                and
                support_hits
            ):

                target_levels = [

                    level

                    for level
                    in resistance_levels

                    if level > c
                ]

                if target_levels:

                    target = min(
                        target_levels
                    )

                    pending = {

                        "side":
                            "LONG",

                        "signal_bar":
                            i,

                        "signal_time":
                            candle[
                                "timestamp"
                            ].isoformat(),

                        "trigger":
                            h,

                        "stop":
                            l,

                        "target":
                            target,

                        "target_source":
                            "nearest_reversal_resistance"
                    }

            # -------------------------------------------------
            # SHORT
            # -------------------------------------------------

            elif c < o:

                resistance_hits = [

                    level

                    for level
                    in resistance_levels

                    if (
                        level > 0
                        and
                        abs(
                            h - level
                        ) / level
                        <= TOUCH_TOLERANCE
                        and
                        c < level
                    )
                ]

                if resistance_hits:

                    target_levels = [

                        level

                        for level
                        in support_levels

                        if level < c
                    ]

                    if target_levels:

                        target = max(
                            target_levels
                        )

                        pending = {

                            "side":
                                "SHORT",

                            "signal_bar":
                                i,

                            "signal_time":
                                candle[
                                    "timestamp"
                                ].isoformat(),

                            "trigger":
                                l,

                            "stop":
                                h,

                            "target":
                                target,

                            "target_source":
                                "nearest_reversal_support"
                        }

    return pd.DataFrame(
        trades
    )


# =========================================================
# SUMMARY
# =========================================================

def create_summary(trades):

    if trades.empty:

        return pd.DataFrame([{

            "mode":
                "LUXALGO_REVERSAL",

            "trades": 0,

            "wins": 0,

            "losses": 0,

            "win_rate_pct": 0,

            "net_pnl_inr": 0,

            "gross_pnl_inr": 0,

            "fees_inr": 0,

            "profit_factor": 0,

            "max_drawdown_inr": 0
        }])

    wins = trades[
        trades["net_pnl_inr"] > 0
    ]

    losses = trades[
        trades["net_pnl_inr"] <= 0
    ]

    gross_profit = float(
        wins["net_pnl_inr"].sum()
    )

    gross_loss = abs(
        float(
            losses["net_pnl_inr"].sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = math.inf

    equity = (
        trades[
            "net_pnl_inr"
        ].cumsum()
    )

    peak = (
        equity.cummax()
    )

    drawdown = (
        equity -
        peak
    )

    return pd.DataFrame([{

        "mode":
            "LUXALGO_REVERSAL",

        "trades":
            len(trades),

        "wins":
            len(wins),

        "losses":
            len(losses),

        "win_rate_pct":
            round(
                len(wins) /
                len(trades) *
                100,
                2
            ),

        "net_pnl_inr":
            round(
                float(
                    trades[
                        "net_pnl_inr"
                    ].sum()
                ),
                2
            ),

        "gross_pnl_inr":
            round(
                float(
                    trades[
                        "gross_pnl_inr"
                    ].sum()
                ),
                2
            ),

        "fees_inr":
            round(
                float(
                    trades[
                        "fees_inr"
                    ].sum()
                ),
                2
            ),

        "profit_factor":
            round(
                profit_factor,
                3
            )
            if math.isfinite(
                profit_factor
            )
            else "INF",

        "max_drawdown_inr":
            round(
                abs(
                    float(
                        drawdown.min()
                    )
                ),
                2
            )
    }])


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
        choices=[
            "LAST_PRICE",
            "MARK_PRICE"
        ],
        default="LAST_PRICE"
    )

    args = parser.parse_args()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    df = download_klines(
        pair=args.pair,
        days=args.days,
        price_type=args.price_type
    )

    # =====================================================
    # BACKTEST
    # =====================================================

    print()
    print("=" * 60)
    print("LUXALGO REVERSAL BACKTEST")
    print("=" * 60)

    print(
        f"Candles       : {len(df):,}"
    )

    print(
        f"Period        : "
        f"{df['timestamp'].iloc[0]} "
        f"-> "
        f"{df['timestamp'].iloc[-1]}"
    )

    print(
        f"Pivot length  : "
        f"{PIVOT_LENGTH}"
    )

    print(
        f"Margin/trade  : "
        f"INR {MARGIN_INR:,.0f}"
    )

    print(
        f"Leverage      : "
        f"{LEVERAGE:g}x"
    )

    print(
        f"Position size : "
        f"INR "
        f"{MARGIN_INR * LEVERAGE:,.0f}"
    )

    print(
        f"Price type    : "
        f"{args.price_type}"
    )

    trades = run_backtest(
        df,
        args.pair.upper()
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    trades_file = (
        output_dir /
        "trades_luxalgo_reversal.csv"
    )

    summary_file = (
        output_dir /
        "summary.csv"
    )

    trades.to_csv(
        trades_file,
        index=False
    )

    summary = create_summary(
        trades
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # =====================================================
    # PRINT
    # =====================================================

    print()
    print("=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Trades saved : "
        f"{trades_file}"
    )

    print(
        f"Summary saved: "
        f"{summary_file}"
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
