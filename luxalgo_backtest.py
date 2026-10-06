import argparse
import math
import time
from pathlib import Path

import pandas as pd
import requests


# =========================================================
# CONFIG
# =========================================================

PIVOT_LENGTH = 50

MARGIN_INR = 1000.0
LEVERAGE = 10.0

TAKER_FEE = 0.00040       # 0.040% per side
GST_ON_FEE = 0.18         # 18% GST on fee
SLIPPAGE = 0.00020        # 0.02% per side

TOUCH_TOLERANCE = 0.0015  # 0.15%

MIN_STRUCTURAL_RR = 1.50
FALLBACK_RR = 2.00

MAX_HOLD_BARS = 2000

API_URL = "https://api.sharkexchange.in/v1/market/klines"

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


# =========================================================
# DATA DOWNLOAD
# =========================================================

def download_shark_data(
    pair="BTCUSDT",
    interval="5m",
    days=30,
    price_type="LAST_PRICE",
):

    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    print("\n======================================")
    print("SHARK EXCHANGE DATA DOWNLOAD")
    print("======================================")

    print(f"Pair       : {pair}")
    print(f"Timeframe  : {interval}")
    print(f"Days       : {days}")
    print(f"Price type : {price_type}")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000

    step = INTERVAL_MS[interval]

    cursor = start_ms
    all_rows = []

    session = requests.Session()

    page = 0

    while cursor < now_ms:

        page += 1

        payload = {
            "pair": pair.upper(),
            "interval": interval,
            "startTime": cursor,
            "endTime": now_ms,
            "limit": 1000,
        }

        print(f"\nDownloading batch {page}...")

        response = session.post(
            f"{API_URL}?priceType={price_type}",
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):

            rows = None

            for key in ["data", "result", "klines", "rows"]:

                if isinstance(data.get(key), list):
                    rows = data[key]
                    break

            if rows is None:
                raise RuntimeError(
                    f"Unexpected API response:\n{data}"
                )

        elif isinstance(data, list):

            rows = data

        else:

            raise RuntimeError(
                f"Unexpected API response type: {type(data)}"
            )

        if not rows:

            print("No more candles.")
            break

        batch = []

        for candle in rows:

            if not isinstance(candle, dict):
                continue

            if "startTime" not in candle:
                continue

            try:

                timestamp = int(candle["startTime"])

                batch.append(
                    {
                        "timestamp": pd.to_datetime(
                            timestamp,
                            unit="ms",
                            utc=True,
                        ),
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(
                            candle.get("volume", 0)
                        ),
                        "endTime": int(
                            candle.get(
                                "endTime",
                                timestamp + step,
                            )
                        ),
                    }
                )

            except Exception:
                continue

        if not batch:
            raise RuntimeError(
                "API returned candles but they could not be parsed."
            )

        all_rows.extend(batch)

        last_timestamp = max(
            x["timestamp"] for x in batch
        )

        next_cursor = (
            int(last_timestamp.timestamp() * 1000)
            + step
        )

        print(
            f"Batch {page}: "
            f"{len(batch)} candles | "
            f"up to {last_timestamp}"
        )

        if next_cursor <= cursor:
            raise RuntimeError(
                "Pagination did not advance."
            )

        cursor = next_cursor

        time.sleep(0.15)

        if len(batch) < 1000:
            break

    if not all_rows:
        raise RuntimeError(
            f"No historical data received for {pair}."
        )

    df = pd.DataFrame(all_rows)

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # Remove current/open candle
    current_ms = int(time.time() * 1000)

    df = df[
        df["endTime"] < current_ms
    ].copy()

    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    if len(df) < 200:

        raise RuntimeError(
            f"Only {len(df)} closed candles downloaded. "
            f"Need at least 200."
        )

    print("\n======================================")
    print("DOWNLOAD COMPLETE")
    print("======================================")

    print(f"Candles : {len(df):,}")
    print(f"From    : {df.timestamp.iloc[0]}")
    print(f"To      : {df.timestamp.iloc[-1]}")

    return df


# =========================================================
# PIVOT FUNCTIONS
# =========================================================

def pivot_high(df, index):

    start = index - PIVOT_LENGTH
    end = index + PIVOT_LENGTH + 1

    if start < 0:
        return False

    if end > len(df):
        return False

    value = df.iloc[index]["high"]

    left = df.iloc[start:index]["high"]
    right = df.iloc[index + 1:end]["high"]

    return (
        value >= left.max()
        and
        value >= right.max()
    )


def pivot_low(df, index):

    start = index - PIVOT_LENGTH
    end = index + PIVOT_LENGTH + 1

    if start < 0:
        return False

    if end > len(df):
        return False

    value = df.iloc[index]["low"]

    left = df.iloc[start:index]["low"]
    right = df.iloc[index + 1:end]["low"]

    return (
        value <= left.min()
        and
        value <= right.min()
    )


# =========================================================
# FEE
# =========================================================

def trading_fee(notional):

    fee = notional * TAKER_FEE

    gst = fee * GST_ON_FEE

    return fee + gst


# =========================================================
# SLIPPAGE
# =========================================================

def entry_price(price, side):

    if side == "LONG":

        return price * (1 + SLIPPAGE)

    return price * (1 - SLIPPAGE)


def exit_price(price, side):

    if side == "LONG":

        return price * (1 - SLIPPAGE)

    return price * (1 + SLIPPAGE)


# =========================================================
# STRUCTURAL TARGET
# =========================================================

def find_structural_target(
    side,
    entry,
    stop,
    levels,
):

    risk = abs(entry - stop)

    if risk <= 0:
        return None

    candidates = []

    if side == "LONG":

        for level in levels:

            if level["type"] != "RESISTANCE":
                continue

            if level["price"] <= entry:
                continue

            rr = (
                level["price"] - entry
            ) / risk

            if rr >= MIN_STRUCTURAL_RR:

                candidates.append(
                    (
                        level["price"],
                        rr,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x[0]
        )

        return candidates[0][0]

    else:

        for level in levels:

            if level["type"] != "SUPPORT":
                continue

            if level["price"] >= entry:
                continue

            rr = (
                entry - level["price"]
            ) / risk

            if rr >= MIN_STRUCTURAL_RR:

                candidates.append(
                    (
                        level["price"],
                        rr,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return candidates[0][0]


# =========================================================
# BACKTEST
# =========================================================

def backtest(
    df,
    symbol,
    fixed_rr=None,
):

    levels = []

    trades = []

    pending = None

    position = None

    running_high = -math.inf
    running_low = math.inf

    running_high_bar = None
    running_low_bar = None

    last_pivot = None

    start = PIVOT_LENGTH * 2

    for i in range(start, len(df)):

        row = df.iloc[i]

        # -------------------------------------------------
        # PIVOT CONFIRMATION
        # -------------------------------------------------

        pivot_index = i - PIVOT_LENGTH

        # ---------------------------
        # PIVOT HIGH
        # ---------------------------

        if pivot_high(
            df,
            pivot_index,
        ):

            price = float(
                df.iloc[pivot_index]["high"]
            )

            # Missed LOW
            if (
                last_pivot == "HIGH"
                and
                running_low != math.inf
            ):

                levels.append(
                    {
                        "price": running_low,
                        "type": "SUPPORT",
                        "source": "MISSED",
                        "bar": running_low_bar,
                    }
                )

            # Regular resistance
            levels.append(
                {
                    "price": price,
                    "type": "RESISTANCE",
                    "source": "REGULAR",
                    "bar": pivot_index,
                }
            )

            last_pivot = "HIGH"

            running_high = price
            running_high_bar = pivot_index

            running_low = float(
                df.iloc[pivot_index]["low"]
            )

            running_low_bar = pivot_index

        # ---------------------------
        # PIVOT LOW
        # ---------------------------

        if pivot_low(
            df,
            pivot_index,
        ):

            price = float(
                df.iloc[pivot_index]["low"]
            )

            # Missed HIGH
            if (
                last_pivot == "LOW"
                and
                running_high != -math.inf
            ):

                levels.append(
                    {
                        "price": running_high,
                        "type": "RESISTANCE",
                        "source": "MISSED",
                        "bar": running_high_bar,
                    }
                )

            # Regular support
            levels.append(
                {
                    "price": price,
                    "type": "SUPPORT",
                    "source": "REGULAR",
                    "bar": pivot_index,
                }
            )

            last_pivot = "LOW"

            running_low = price
            running_low_bar = pivot_index

            running_high = float(
                df.iloc[pivot_index]["high"]
            )

            running_high_bar = pivot_index

        # -------------------------------------------------
        # UPDATE EXTREMES
        # -------------------------------------------------

        high = float(row["high"])
        low = float(row["low"])

        if high > running_high:

            running_high = high
            running_high_bar = i

        if low < running_low:

            running_low = low
            running_low_bar = i

        # -------------------------------------------------
        # MANAGE OPEN POSITION
        # -------------------------------------------------

        if position is not None:

            side = position["side"]

            stop = position["stop"]

            target = position["target"]

            bars_held = (
                i - position["entry_bar"]
            )

            hit_sl = False
            hit_target = False

            if side == "LONG":

                hit_sl = low <= stop

                hit_target = (
                    high >= target
                )

            else:

                hit_sl = high >= stop

                hit_target = (
                    low <= target
                )

            exit_reason = None
            raw_exit = None

            # Conservative rule:
            # if both happen in same candle,
            # assume SL happens first.

            if hit_sl:

                exit_reason = "SL"
                raw_exit = stop

            elif hit_target:

                exit_reason = "TARGET"
                raw_exit = target

            elif bars_held >= MAX_HOLD_BARS:

                exit_reason = "TIME"
                raw_exit = float(
                    row["close"]
                )

            if exit_reason:

                actual_exit = exit_price(
                    raw_exit,
                    side,
                )

                qty = position["qty"]

                if side == "LONG":

                    gross = (
                        actual_exit
                        - position["entry"]
                    ) * qty

                else:

                    gross = (
                        position["entry"]
                        - actual_exit
                    ) * qty

                exit_notional = (
                    actual_exit * qty
                )

                fees = (
                    position["entry_fee"]
                    +
                    trading_fee(
                        exit_notional
                    )
                )

                net = gross - fees

                trades.append(
                    {
                        "symbol": symbol,
                        "side": side,

                        "signal_time":
                            df.iloc[
                                position["signal_bar"]
                            ]["timestamp"],

                        "entry_time":
                            df.iloc[
                                position["entry_bar"]
                            ]["timestamp"],

                        "exit_time":
                            row["timestamp"],

                        "entry":
                            position["entry"],

                        "stop":
                            stop,

                        "target":
                            target,

                        "exit":
                            actual_exit,

                        "qty":
                            qty,

                        "margin":
                            MARGIN_INR,

                        "leverage":
                            LEVERAGE,

                        "gross_pnl":
                            gross,

                        "fees":
                            fees,

                        "net_pnl":
                            net,

                        "planned_rr":
                            position["planned_rr"],

                        "exit_reason":
                            exit_reason,

                        "level_source":
                            position[
                                "level_source"
                            ],

                        "level_price":
                            position[
                                "level_price"
                            ],
                    }
                )

                position = None

                pending = None

                continue

        # -------------------------------------------------
        # PENDING ENTRY
        # -------------------------------------------------

        if (
            position is None
            and
            pending is not None
        ):

            if i > pending["signal_bar"]:

                high = float(row["high"])
                low = float(row["low"])

                # ---------------------------
                # LONG BREAKOUT
                # ---------------------------

                if (
                    pending["side"] == "LONG"
                    and
                    high > pending["trigger"]
                ):

                    raw_entry = (
                        pending["trigger"]
                    )

                    actual_entry = entry_price(
                        raw_entry,
                        "LONG",
                    )

                    stop = pending["stop"]

                    risk = (
                        actual_entry - stop
                    )

                    if risk > 0:

                        target = (
                            pending["target"]
                        )

                        if target > actual_entry:

                            planned_rr = (
                                target
                                - actual_entry
                            ) / risk

                            qty = (
                                MARGIN_INR
                                * LEVERAGE
                            ) / actual_entry

                            entry_fee = (
                                trading_fee(
                                    actual_entry
                                    * qty
                                )
                            )

                            position = {
                                "side": "LONG",
                                "entry":
                                    actual_entry,
                                "stop":
                                    stop,
                                "target":
                                    target,
                                "qty":
                                    qty,
                                "entry_fee":
                                    entry_fee,
                                "entry_bar":
                                    i,
                                "signal_bar":
                                    pending[
                                        "signal_bar"
                                    ],
                                "planned_rr":
                                    planned_rr,
                                "level_source":
                                    pending[
                                        "level_source"
                                    ],
                                "level_price":
                                    pending[
                                        "level_price"
                                    ],
                            }

                            pending = None

                # ---------------------------
                # SHORT BREAKDOWN
                # ---------------------------

                elif (
                    pending["side"] == "SHORT"
                    and
                    low < pending["trigger"]
                ):

                    raw_entry = (
                        pending["trigger"]
                    )

                    actual_entry = entry_price(
                        raw_entry,
                        "SHORT",
                    )

                    stop = pending["stop"]

                    risk = (
                        stop - actual_entry
                    )

                    if risk > 0:

                        target = pending["target"]

                        if target < actual_entry:

                            planned_rr = (
                                actual_entry
                                - target
                            ) / risk

                            qty = (
                                MARGIN_INR
                                * LEVERAGE
                            ) / actual_entry

                            entry_fee = (
                                trading_fee(
                                    actual_entry
                                    * qty
                                )
                            )

                            position = {
                                "side": "SHORT",
                                "entry":
                                    actual_entry,
                                "stop":
                                    stop,
                                "target":
                                    target,
                                "qty":
                                    qty,
                                "entry_fee":
                                    entry_fee,
                                "entry_bar":
                                    i,
                                "signal_bar":
                                    pending[
                                        "signal_bar"
                                    ],
                                "planned_rr":
                                    planned_rr,
                                "level_source":
                                    pending[
                                        "level_source"
                                    ],
                                "level_price":
                                    pending[
                                        "level_price"
                                    ],
                            }

                            pending = None

        # -------------------------------------------------
        # NEW SIGNAL
        # -------------------------------------------------

        if (
            position is None
            and
            pending is None
        ):

            open_price = float(
                row["open"]
            )

            close_price = float(
                row["close"]
            )

            high = float(row["high"])
            low = float(row["low"])

            bullish = (
                close_price > open_price
            )

            bearish = (
                close_price < open_price
            )

            # ---------------------------
            # SUPPORT
            # ---------------------------

            supports = []

            for level in levels:

                if level["type"] != "SUPPORT":
                    continue

                distance = abs(
                    low - level["price"]
                ) / level["price"]

                if (
                    distance
                    <= TOUCH_TOLERANCE
                    and
                    close_price
                    > level["price"]
                ):

                    supports.append(level)

            # ---------------------------
            # RESISTANCE
            # ---------------------------

            resistances = []

            for level in levels:

                if level["type"] != "RESISTANCE":
                    continue

                distance = abs(
                    high - level["price"]
                ) / level["price"]

                if (
                    distance
                    <= TOUCH_TOLERANCE
                    and
                    close_price
                    < level["price"]
                ):

                    resistances.append(level)

            # ---------------------------
            # LONG SETUP
            # ---------------------------

            if bullish and supports:

                level = max(
                    supports,
                    key=lambda x:
                    x["price"],
                )

                stop = low

                if stop < close_price:

                    if fixed_rr is not None:

                        target = (
                            close_price
                            +
                            (
                                close_price
                                - stop
                            )
                            * fixed_rr
                        )

                    else:

                        target = (
                            find_structural_target(
                                "LONG",
                                close_price,
                                stop,
                                levels,
                            )
                        )

                        if target is None:

                            target = (
                                close_price
                                +
                                (
                                    close_price
                                    - stop
                                )
                                * FALLBACK_RR
                            )

                    pending = {
                        "side": "LONG",
                        "signal_bar": i,
                        "trigger": high,
                        "stop": stop,
                        "target": target,
                        "level_source":
                            level["source"],
                        "level_price":
                            level["price"],
                    }

            # ---------------------------
            # SHORT SETUP
            # ---------------------------

            elif (
                bearish
                and
                resistances
            ):

                level = min(
                    resistances,
                    key=lambda x:
                    x["price"],
                )

                stop = high

                if stop > close_price:

                    if fixed_rr is not None:

                        target = (
                            close_price
                            -
                            (
                                stop
                                - close_price
                            )
                            * fixed_rr
                        )

                    else:

                        target = (
                            find_structural_target(
                                "SHORT",
                                close_price,
                                stop,
                                levels,
                            )
                        )

                        if target is None:

                            target = (
                                close_price
                                -
                                (
                                    stop
                                    - close_price
                                )
                                * FALLBACK_RR
                            )

                    pending = {
                        "side": "SHORT",
                        "signal_bar": i,
                        "trigger": low,
                        "stop": stop,
                        "target": target,
                        "level_source":
                            level["source"],
                        "level_price":
                            level["price"],
                    }

    return pd.DataFrame(trades)


# =========================================================
# SUMMARY
# =========================================================

def make_summary(
    trades,
    mode,
):

    if trades.empty:

        return {
            "mode": mode,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "gross_pnl": 0,
            "fees": 0,
            "net_pnl": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
        }

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

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = float("inf")

    equity = (
        trades["net_pnl"]
        .cumsum()
    )

    peak = equity.cummax()

    drawdown = (
        equity - peak
    )

    return {
        "mode": mode,

        "trades":
            len(trades),

        "wins":
            len(wins),

        "losses":
            len(losses),

        "win_rate":
            round(
                len(wins)
                / len(trades)
                * 100,
                2,
            ),

        "gross_pnl":
            round(
                float(
                    trades[
                        "gross_pnl"
                    ].sum()
                ),
                2,
            ),

        "fees":
            round(
                float(
                    trades[
                        "fees"
                    ].sum()
                ),
                2,
            ),

        "net_pnl":
            round(
                float(
                    trades[
                        "net_pnl"
                    ].sum()
                ),
                2,
            ),

        "profit_factor":
            (
                round(
                    profit_factor,
                    3,
                )
                if math.isfinite(
                    profit_factor
                )
                else "INF"
            ),

        "max_drawdown":
            round(
                abs(
                    float(
                        drawdown.min()
                    )
                ),
                2,
            ),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair",
        default="BTCUSDT",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--price-type",
        choices=[
            "LAST_PRICE",
            "MARK_PRICE",
        ],
        default="LAST_PRICE",
    )

    parser.add_argument(
        "--output",
        default="results",
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    df = download_shark_data(
        pair=args.pair,
        interval="5m",
        days=args.days,
        price_type=args.price_type,
    )

    # Save downloaded candles
    data_file = (
        output
        / f"{args.pair}_5m_data.csv"
    )

    df.to_csv(
        data_file,
        index=False,
    )

    # -----------------------------------------------------
    # RUN BACKTESTS
    # -----------------------------------------------------

    print("\n======================================")
    print("BACKTEST")
    print("======================================")

    print(
        f"Margin/trade : ₹{MARGIN_INR:,.0f}"
    )

    print(
        f"Leverage     : {LEVERAGE}x"
    )

    print(
        f"Position size: ₹{MARGIN_INR * LEVERAGE:,.0f}"
    )

    summaries = []

    # Structural target
    structural = backtest(
        df,
        args.pair,
        fixed_rr=None,
    )

    structural.to_csv(
        output
        / "trades_structural.csv",
        index=False,
    )

    summaries.append(
        make_summary(
            structural,
            "STRUCTURAL",
        )
    )

    # Fixed RR tests
    for rr in [
        1.0,
        1.5,
        2.0,
        3.0,
    ]:

        print(
            f"\nRunning {rr}:1..."
        )

        trades = backtest(
            df,
            args.pair,
            fixed_rr=rr,
        )

        trades.to_csv(
            output
            / f"trades_RR_{rr}.csv",
            index=False,
        )

        summaries.append(
            make_summary(
                trades,
                f"{rr}:1",
            )
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary = pd.DataFrame(
        summaries
    )

    summary.to_csv(
        output / "summary.csv",
        index=False,
    )

    print("\n======================================")
    print("FINAL RESULT")
    print("======================================")

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nResults saved in:",
        output.resolve(),
    )


if __name__ == "__main__":
    main()
