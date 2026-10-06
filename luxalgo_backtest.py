#!/usr/bin/env python3

import argparse
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


# =========================================================
# CONFIG
# =========================================================

SHARK_URL = "https://api.sharkexchange.in/v1/market/klines"

PIVOT_LENGTH = 50

MARGIN_INR = 1000.0
LEVERAGE = 10.0

TOUCH_TOLERANCE_PCT = 0.0015

TAKER_FEE = 0.00040
GST_ON_FEE = 0.18
SLIPPAGE = 0.00020

MAX_LIMIT = 1000
MAX_BARS_IN_TRADE = 2000


INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class Level:
    price: float
    kind: str
    source: str
    bar: int
    active: bool = True


@dataclass
class Trade:
    symbol: str
    side: str

    signal_bar: int
    entry_bar: int
    exit_bar: int

    signal_time: str
    entry_time: str
    exit_time: str

    entry: float
    stop: float
    target: float
    exit: float

    qty: float
    margin: float
    leverage: float

    pnl_gross: float
    fees: float
    pnl_net: float

    rr_planned: float

    exit_reason: str

    level_source: str
    level_price: float


# =========================================================
# SHARK API
# =========================================================

def extract_rows(payload):

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):

        for key in (
            "data",
            "result",
            "klines",
            "rows"
        ):

            value = payload.get(key)

            if isinstance(value, list):
                return value

    raise RuntimeError(
        f"Unexpected Shark API response: "
        f"{type(payload).__name__}"
    )


def normalize_rows(rows):

    output = []

    for row in rows:

        if not isinstance(row, dict):
            continue

        if "startTime" not in row:
            continue

        try:

            start_time = int(row["startTime"])

            output.append({
                "timestamp": pd.to_datetime(
                    start_time,
                    unit="ms",
                    utc=True
                ),

                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),

                "volume": float(
                    row.get("volume", 0)
                ),

                "endTime": int(
                    row.get(
                        "endTime",
                        start_time
                    )
                )
            })

        except Exception:
            continue

    return output


def download_shark_data(
    pair,
    interval,
    days,
    price_type
):

    if interval not in INTERVAL_MS:

        raise ValueError(
            f"Unsupported interval: {interval}"
        )

    interval_ms = INTERVAL_MS[interval]

    now = datetime.now(timezone.utc)

    start = (
        now -
        timedelta(days=days)
    )

    start_ms = int(
        start.timestamp() * 1000
    )

    end_ms = int(
        now.timestamp() * 1000
    )

    cursor = start_ms

    all_rows = []

    page = 0

    session = requests.Session()

    session.headers.update({
        "Content-Type": "application/json",
        "User-Agent": "LuxAlgo-Backtest/1.0"
    })

    print()
    print("=" * 60)
    print("DOWNLOADING SHARK EXCHANGE DATA")
    print("=" * 60)

    print(f"Pair       : {pair}")
    print(f"Interval   : {interval}")
    print(f"Days       : {days}")
    print(f"Price type : {price_type}")
    print()

    while cursor < end_ms:

        page += 1

        body = {
            "pair": pair.upper(),
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_LIMIT
        }

        try:

            response = session.post(
                f"{SHARK_URL}?priceType={price_type}",
                json=body,
                timeout=30
            )

            response.raise_for_status()

            payload = response.json()

            rows = extract_rows(payload)

        except Exception as e:

            print(
                f"Page {page} ERROR: {e}"
            )

            time.sleep(2)

            page -= 1

            continue

        if not rows:

            print(
                f"Page {page}: no more candles."
            )

            break

        normalized = normalize_rows(rows)

        if not normalized:

            raise RuntimeError(
                "Shark returned candle data "
                "but it could not be parsed."
            )

        all_rows.extend(normalized)

        latest_timestamp = max(
            row["timestamp"]
            for row in normalized
        )

        print(
            f"Page {page}: "
            f"{len(normalized)} candles | "
            f"through "
            f"{latest_timestamp.isoformat()}"
        )

        latest_ms = int(
            latest_timestamp.timestamp() * 1000
        )

        next_cursor = (
            latest_ms +
            interval_ms
        )

        if next_cursor <= cursor:

            raise RuntimeError(
                "Pagination did not advance."
            )

        cursor = next_cursor

        time.sleep(0.15)

        # IMPORTANT:
        # Do NOT stop just because this page has fewer
        # than 1000 candles.
        #
        # This prevents the previous 999-candle bug.

        if latest_ms >= end_ms:
            break

    if not all_rows:

        raise RuntimeError(
            f"No candles received for {pair}."
        )

    df = pd.DataFrame(
        all_rows
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Remove future/open candle
    # -----------------------------------------------------

    current_ms = int(
        time.time() * 1000
    )

    df = df[
        df["endTime"] < current_ms
    ].copy()

    # -----------------------------------------------------
    # Keep requested period
    # -----------------------------------------------------

    start_timestamp = pd.Timestamp(
        start
    )

    end_timestamp = pd.Timestamp(
        now
    )

    df = df[
        (df["timestamp"] >= start_timestamp) &
        (df["timestamp"] <= end_timestamp)
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    if len(df) < 200:

        raise RuntimeError(
            f"Only {len(df)} closed candles "
            f"were downloaded.\n"
            f"Need at least 200 candles."
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
# INDICATOR / PIVOT FUNCTIONS
# =========================================================

def is_pivot_high(
    df,
    pivot_i,
    length
):

    price = float(
        df.at[pivot_i, "high"]
    )

    left = df[
        "high"
    ].iloc[
        pivot_i - length:pivot_i
    ]

    right = df[
        "high"
    ].iloc[
        pivot_i + 1:
        pivot_i + length + 1
    ]

    return (
        price >= float(left.max())
        and
        price >= float(right.max())
    )


def is_pivot_low(
    df,
    pivot_i,
    length
):

    price = float(
        df.at[pivot_i, "low"]
    )

    left = df[
        "low"
    ].iloc[
        pivot_i - length:pivot_i
    ]

    right = df[
        "low"
    ].iloc[
        pivot_i + 1:
        pivot_i + length + 1
    ]

    return (
        price <= float(left.min())
        and
        price <= float(right.min())
    )


# =========================================================
# FEES / SLIPPAGE
# =========================================================

def fee_for_notional(
    notional
):

    trading_fee = (
        notional *
        TAKER_FEE
    )

    return (
        trading_fee *
        (1 + GST_ON_FEE)
    )


def apply_entry_slippage(
    price,
    side
):

    if side == "LONG":

        return price * (
            1 + SLIPPAGE
        )

    return price * (
        1 - SLIPPAGE
    )


def apply_exit_slippage(
    price,
    side
):

    if side == "LONG":

        return price * (
            1 - SLIPPAGE
        )

    return price * (
        1 + SLIPPAGE
    )


# =========================================================
# LUXALGO REVERSAL TARGET
# =========================================================

def find_reversal_target(
    side,
    entry,
    levels
):

    # LONG:
    # nearest resistance above entry

    if side == "LONG":

        candidates = [
            level

            for level in levels

            if (
                level.active
                and
                level.kind == "resistance"
                and
                level.price > entry
            )
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda x: x.price
        )

    # SHORT:
    # nearest support below entry

    if side == "SHORT":

        candidates = [
            level

            for level in levels

            if (
                level.active
                and
                level.kind == "support"
                and
                level.price < entry
            )
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda x: x.price
        )

    return None


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(
    df,
    symbol
):

    levels = []

    trades = []

    pending = None

    open_trade = None

    running_high = -math.inf
    running_low = math.inf

    running_high_bar = None
    running_low_bar = None

    last_pivot_type = None

    start = (
        PIVOT_LENGTH * 2
    )

    n = len(df)

    for i in range(
        start,
        n
    ):

        row = df.iloc[i]

        # =================================================
        # CONFIRM OLD PIVOT
        # =================================================

        pivot_i = (
            i -
            PIVOT_LENGTH
        )

        # -------------------------------------------------
        # PIVOT HIGH
        # -------------------------------------------------

        if is_pivot_high(
            df,
            pivot_i,
            PIVOT_LENGTH
        ):

            pivot_price = float(
                df.at[
                    pivot_i,
                    "high"
                ]
            )

            # Missed support
            if (
                last_pivot_type == "high"
                and
                running_low < math.inf
            ):

                levels.append(
                    Level(
                        price=running_low,
                        kind="support",
                        source="missed",
                        bar=int(
                            running_low_bar
                        )
                    )
                )

            # Regular resistance
            levels.append(
                Level(
                    price=pivot_price,
                    kind="resistance",
                    source="regular",
                    bar=pivot_i
                )
            )

            last_pivot_type = "high"

            running_high = pivot_price
            running_high_bar = pivot_i

            running_low = float(
                df.at[
                    pivot_i,
                    "low"
                ]
            )

            running_low_bar = pivot_i

        # -------------------------------------------------
        # PIVOT LOW
        # -------------------------------------------------

        if is_pivot_low(
            df,
            pivot_i,
            PIVOT_LENGTH
        ):

            pivot_price = float(
                df.at[
                    pivot_i,
                    "low"
                ]
            )

            # Missed resistance
            if (
                last_pivot_type == "low"
                and
                running_high > -math.inf
            ):

                levels.append(
                    Level(
                        price=running_high,
                        kind="resistance",
                        source="missed",
                        bar=int(
                            running_high_bar
                        )
                    )
                )

            # Regular support
            levels.append(
                Level(
                    price=pivot_price,
                    kind="support",
                    source="regular",
                    bar=pivot_i
                )
            )

            last_pivot_type = "low"

            running_low = pivot_price
            running_low_bar = pivot_i

            running_high = float(
                df.at[
                    pivot_i,
                    "high"
                ]
            )

            running_high_bar = pivot_i

        # =================================================
        # UPDATE EXTREMES
        # =================================================

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        if high > running_high:

            running_high = high
            running_high_bar = i

        if low < running_low:

            running_low = low
            running_low_bar = i

        # =================================================
        # MANAGE OPEN TRADE
        # =================================================

        if open_trade is not None:

            trade = open_trade

            bars_held = (
                i -
                trade["entry_bar"]
            )

            hi = float(
                row["high"]
            )

            lo = float(
                row["low"]
            )

            if trade["side"] == "LONG":

                hit_sl = (
                    lo <= trade["stop"]
                )

                hit_target = (
                    hi >= trade["target"]
                )

            else:

                hit_sl = (
                    hi >= trade["stop"]
                )

                hit_target = (
                    lo <= trade["target"]
                )

            exit_reason = None
            raw_exit = None

            # SL gets priority if both happen
            # in the same candle.

            if hit_sl:

                exit_reason = "SL"
                raw_exit = trade["stop"]

            elif hit_target:

                exit_reason = "TARGET"
                raw_exit = trade["target"]

            elif (
                bars_held >=
                MAX_BARS_IN_TRADE
            ):

                exit_reason = "TIME"
                raw_exit = float(
                    row["close"]
                )

            if exit_reason:

                exit_price = apply_exit_slippage(
                    raw_exit,
                    trade["side"]
                )

                if trade["side"] == "LONG":

                    gross = (
                        exit_price -
                        trade["entry"]
                    ) * trade["qty"]

                else:

                    gross = (
                        trade["entry"] -
                        exit_price
                    ) * trade["qty"]

                exit_notional = (
                    exit_price *
                    trade["qty"]
                )

                exit_fee = fee_for_notional(
                    exit_notional
                )

                total_fees = (
                    trade["entry_fee"] +
                    exit_fee
                )

                net = (
                    gross -
                    total_fees
                )

                trades.append(
                    Trade(
                        symbol=symbol,

                        side=trade["side"],

                        signal_bar=trade[
                            "signal_bar"
                        ],

                        entry_bar=trade[
                            "entry_bar"
                        ],

                        exit_bar=i,

                        signal_time=df.at[
                            trade["signal_bar"],
                            "timestamp"
                        ].isoformat(),

                        entry_time=df.at[
                            trade["entry_bar"],
                            "timestamp"
                        ].isoformat(),

                        exit_time=df.at[
                            i,
                            "timestamp"
                        ].isoformat(),

                        entry=trade["entry"],

                        stop=trade["stop"],

                        target=trade["target"],

                        exit=exit_price,

                        qty=trade["qty"],

                        margin=MARGIN_INR,

                        leverage=LEVERAGE,

                        pnl_gross=gross,

                        fees=total_fees,

                        pnl_net=net,

                        rr_planned=trade[
                            "rr_planned"
                        ],

                        exit_reason=exit_reason,

                        level_source=trade[
                            "level_source"
                        ],

                        level_price=trade[
                            "level_price"
                        ]
                    )
                )

                open_trade = None
                pending = None

                continue

        # =================================================
        # WAIT FOR ENTRY BREAKOUT
        # =================================================

        if (
            open_trade is None
            and
            pending is not None
        ):

            if (
                i >
                pending["signal_bar"]
            ):

                high = float(
                    row["high"]
                )

                low = float(
                    row["low"]
                )

                # -------------------------------------------------
                # LONG BREAKOUT
                # -------------------------------------------------

                if (
                    pending["side"]
                    == "LONG"
                    and
                    high >
                    pending["trigger"]
                ):

                    entry_raw = (
                        pending["trigger"]
                    )

                    entry = apply_entry_slippage(
                        entry_raw,
                        "LONG"
                    )

                    stop = (
                        pending["stop"]
                    )

                    risk = (
                        entry -
                        stop
                    )

                    target = (
                        pending["target"]
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

                        if rr >= 0.5:

                            qty = (
                                MARGIN_INR *
                                LEVERAGE
                            ) / entry

                            entry_fee = fee_for_notional(
                                entry * qty
                            )

                            open_trade = {

                                "side": "LONG",

                                "entry": entry,

                                "stop": stop,

                                "target": target,

                                "qty": qty,

                                "entry_fee": entry_fee,

                                "entry_bar": i,

                                "signal_bar":
                                    pending[
                                        "signal_bar"
                                    ],

                                "rr_planned": rr,

                                "level_source":
                                    pending[
                                        "level_source"
                                    ],

                                "level_price":
                                    pending[
                                        "level_price"
                                    ]
                            }

                            pending = None

                # -------------------------------------------------
                # SHORT BREAKOUT
                # -------------------------------------------------

                elif (
                    pending["side"]
                    == "SHORT"
                    and
                    low <
                    pending["trigger"]
                ):

                    entry_raw = (
                        pending["trigger"]
                    )

                    entry = apply_entry_slippage(
                        entry_raw,
                        "SHORT"
                    )

                    stop = (
                        pending["stop"]
                    )

                    risk = (
                        stop -
                        entry
                    )

                    target = (
                        pending["target"]
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

                        if rr >= 0.5:

                            qty = (
                                MARGIN_INR *
                                LEVERAGE
                            ) / entry

                            entry_fee = fee_for_notional(
                                entry * qty
                            )

                            open_trade = {

                                "side": "SHORT",

                                "entry": entry,

                                "stop": stop,

                                "target": target,

                                "qty": qty,

                                "entry_fee": entry_fee,

                                "entry_bar": i,

                                "signal_bar":
                                    pending[
                                        "signal_bar"
                                    ],

                                "rr_planned": rr,

                                "level_source":
                                    pending[
                                        "level_source"
                                    ],

                                "level_price":
                                    pending[
                                        "level_price"
                                    ]
                            }

                            pending = None

        # =================================================
        # CREATE NEW SIGNAL
        # =================================================

        if (
            open_trade is None
            and
            pending is None
        ):

            open_price = float(
                row["open"]
            )

            close_price = float(
                row["close"]
            )

            high = float(
                row["high"]
            )

            low = float(
                row["low"]
            )

            bullish = (
                close_price >
                open_price
            )

            bearish = (
                close_price <
                open_price
            )

            # -------------------------------------------------
            # SUPPORT TEST
            # -------------------------------------------------

            support_candidates = [

                level

                for level in levels

                if (
                    level.active
                    and
                    level.kind ==
                    "support"
                    and
                    abs(
                        low -
                        level.price
                    ) /
                    level.price
                    <=
                    TOUCH_TOLERANCE_PCT
                    and
                    close_price >
                    level.price
                )
            ]

            # -------------------------------------------------
            # RESISTANCE TEST
            # -------------------------------------------------

            resistance_candidates = [

                level

                for level in levels

                if (
                    level.active
                    and
                    level.kind ==
                    "resistance"
                    and
                    abs(
                        high -
                        level.price
                    ) /
                    level.price
                    <=
                    TOUCH_TOLERANCE_PCT
                    and
                    close_price <
                    level.price
                )
            ]

            # -------------------------------------------------
            # LONG
            # -------------------------------------------------

            if (
                bullish
                and
                support_candidates
            ):

                level = max(
                    support_candidates,
                    key=lambda x:
                    x.price
                )

                stop = low

                target_level = (
                    find_reversal_target(
                        "LONG",
                        close_price,
                        levels
                    )
                )

                if (
                    stop < close_price
                    and
                    target_level is not None
                    and
                    target_level.price >
                    close_price
                ):

                    pending = {

                        "side": "LONG",

                        "signal_bar": i,

                        "trigger": high,

                        "stop": stop,

                        "target":
                            target_level.price,

                        "level_source":
                            level.source,

                        "level_price":
                            level.price
                    }

            # -------------------------------------------------
            # SHORT
            # -------------------------------------------------

            elif (
                bearish
                and
                resistance_candidates
            ):

                level = min(
                    resistance_candidates,
                    key=lambda x:
                    x.price
                )

                stop = high

                target_level = (
                    find_reversal_target(
                        "SHORT",
                        close_price,
                        levels
                    )
                )

                if (
                    stop > close_price
                    and
                    target_level is not None
                    and
                    target_level.price <
                    close_price
                ):

                    pending = {

                        "side": "SHORT",

                        "signal_bar": i,

                        "trigger": low,

                        "stop": stop,

                        "target":
                            target_level.price,

                        "level_source":
                            level.source,

                        "level_price":
                            level.price
                    }

    return pd.DataFrame(
        [
            asdict(trade)
            for trade in trades
        ]
    )


# =========================================================
# SUMMARY
# =========================================================

def make_summary(
    trades,
    label
):

    if trades.empty:

        return {

            "mode": label,

            "trades": 0,

            "wins": 0,

            "losses": 0,

            "win_rate_pct": 0,

            "net_pnl_inr": 0,

            "gross_pnl_inr": 0,

            "fees_inr": 0,

            "profit_factor": 0,

            "max_drawdown_inr": 0
        }

    wins = trades[
        trades["pnl_net"] > 0
    ]

    losses = trades[
        trades["pnl_net"] <= 0
    ]

    gross_profit = float(
        wins["pnl_net"].sum()
    )

    gross_loss = abs(
        float(
            losses["pnl_net"].sum()
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
        trades["pnl_net"]
        .cumsum()
    )

    peak = (
        equity
        .cummax()
    )

    drawdown = (
        equity -
        peak
    )

    return {

        "mode": label,

        "trades": int(
            len(trades)
        ),

        "wins": int(
            len(wins)
        ),

        "losses": int(
            len(losses)
        ),

        "win_rate_pct": round(
            100 *
            len(wins) /
            len(trades),
            2
        ),

        "net_pnl_inr": round(
            float(
                trades[
                    "pnl_net"
                ].sum()
            ),
            2
        ),

        "gross_pnl_inr": round(
            float(
                trades[
                    "pnl_gross"
                ].sum()
            ),
            2
        ),

        "fees_inr": round(
            float(
                trades[
                    "fees"
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
    }


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
        "--interval",
        default="5m"
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

    parser.add_argument(
        "--out",
        default="results"
    )

    args = parser.parse_args()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    df = download_shark_data(
        pair=args.pair,
        interval=args.interval,
        days=args.days,
        price_type=args.price_type
    )

    # =====================================================
    # BACKTEST
    # =====================================================

    print()
    print("=" * 60)
    print("STARTING LUXALGO BACKTEST")
    print("=" * 60)

    print(
        f"Candles        : {len(df):,}"
    )

    print(
        f"Period         : "
        f"{df['timestamp'].iloc[0]} "
        f"-> "
        f"{df['timestamp'].iloc[-1]}"
    )

    print(
        f"Pivot length   : "
        f"{PIVOT_LENGTH}"
    )

    print(
        f"Margin/trade   : "
        f"INR {MARGIN_INR:,.0f}"
    )

    print(
        f"Leverage       : "
        f"{LEVERAGE:g}x"
    )

    print(
        f"Position size  : "
        f"INR "
        f"{MARGIN_INR * LEVERAGE:,.0f}"
    )

    print(
        f"Price type     : "
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
        args.out
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

    result = pd.DataFrame([
        make_summary(
            trades,
            "LUXALGO_REVERSAL"
        )
    ])

    result.to_csv(
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
        result.to_string(
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
