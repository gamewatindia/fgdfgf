"""
Pivot High/Low (LuxAlgo-style) Backtest
ALL TIMEFRAMES: 5m, 15m, 30m, 1h, 4h

Strategy:
- Confirmed pivot LOW -> LONG
- Confirmed pivot HIGH -> SHORT
- Entry = close of confirmation candle

POSITION:
- Margin = Rs 1000
- Leverage = 10x
- Total position = Rs 10000

75% POSITION:
- Fixed TP = +2% price movement

25% RUNNER:
- After 75% TP -> SL moves to breakeven
- Then closes at:
  - Breakeven
  - Opposite pivot
  - Liquidation
  - End of data

Optional original SL:
- --sl 4 = 4% price movement against position
- Applies only before 2% TP
"""

import argparse
import os
import time
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

MARGIN = 1000.0
LEV = 10.0

MMR = 0.005
FEE_PCT = 0.10

SL = 0.0

# 75% closes at +2%
TP_PCT = 2.0
TP_PART = 0.75
RUNNER_PART = 0.25

TFS = [
    "5m",
    "15m",
    "30m",
    "1h",
    "4h"
]


# =========================================================
# EXCHANGES
# =========================================================

EXCHANGES = {
    "LAST_PRICE": [
        "binanceus",
        "gateio",
        "mexc",
        "kucoin",
        "bitget",
        "okx",
        "bybit"
    ],

    "MARK_PRICE": [
        "gateio",
        "bitget",
        "okx",
        "bybit",
        "binanceusdm",
        "kucoinfutures"
    ],

    "INDEX_PRICE": [
        "gateio",
        "bitget",
        "okx",
        "bybit",
        "binanceusdm"
    ]
}


# =========================================================
# NORMALIZE PAIR
# =========================================================

def norm_pair(pair):

    pair = pair.upper()
    pair = pair.replace("-", "/")
    pair = pair.replace("_", "/")

    if "/" not in pair:

        for quote in ("USDT", "USDC", "USD"):

            if pair.endswith(quote):

                return (
                    pair[:-len(quote)]
                    + "/"
                    + quote
                )

    return pair


# =========================================================
# FETCH DATA
# =========================================================

def fetch(pair, tf, days, warmup, price, first=""):

    import ccxt

    errors = []

    order = (
        [first] if first else []
    ) + [
        e for e in EXCHANGES[price]
        if e != first
    ]

    for exchange_name in order:

        try:

            exchange = getattr(
                ccxt,
                exchange_name
            )({
                "enableRateLimit": True
            })

            exchange.load_markets()

            if price == "LAST_PRICE":

                symbol = pair

            else:

                symbol = (
                    pair
                    + ":"
                    + pair.split("/")[1]
                )

            if symbol not in exchange.markets:

                raise ValueError(
                    f"{symbol} not listed"
                )

            if price == "LAST_PRICE":

                fetch_function = (
                    exchange.fetch_ohlcv
                )

            elif price == "MARK_PRICE":

                fetch_function = (
                    exchange.fetch_mark_ohlcv
                )

            else:

                fetch_function = (
                    exchange.fetch_index_ohlcv
                )

            timeframe_ms = (
                exchange.parse_timeframe(tf)
                * 1000
            )

            now = exchange.milliseconds()

            since = (
                now
                - days * 86400000
                - warmup * timeframe_ms
            )

            rows = []

            while since < now:

                candles = fetch_function(
                    symbol,
                    tf,
                    since=since,
                    limit=1000
                )

                if not candles:
                    break

                rows.extend(candles)

                next_since = (
                    candles[-1][0]
                    + timeframe_ms
                )

                if next_since <= since:
                    break

                since = next_since

                time.sleep(
                    exchange.rateLimit / 1000
                )

            if len(rows) < 2 * warmup:

                raise ValueError(
                    f"only {len(rows)} candles"
                )

            df = pd.DataFrame(
                rows
            ).iloc[:, :5]

            df.columns = [
                "time",
                "open",
                "high",
                "low",
                "close"
            ]

            df["time"] = pd.to_datetime(
                df["time"],
                unit="ms"
            )

            df = (
                df
                .drop_duplicates("time")
                .sort_values("time")
                .reset_index(drop=True)
            )

            print(
                f"[{tf}] data: "
                f"{exchange_name} "
                f"{symbol} "
                f"{price} "
                f"({len(df)} candles)"
            )

            return df

        except Exception as error:

            errors.append(
                f"{exchange_name}: "
                f"{str(error)[:120]}"
            )

    raise RuntimeError(
        " | ".join(errors)
    )


# =========================================================
# BACKTEST
# =========================================================

def backtest(df, length, days, tf):

    high = df["high"].to_numpy(
        dtype=float
    )

    low = df["low"].to_numpy(
        dtype=float
    )

    close = df["close"].to_numpy(
        dtype=float
    )

    times = df["time"]

    cutoff = (
        times.iloc[-1]
        - pd.Timedelta(days=days)
    )

    trades = []

    pivots = 0

    position = 0

    entry_price = None
    entry_time = None

    trade_id = 0

    tp_hit = False

    total_notional = (
        MARGIN * LEV
    )

    tp_notional = (
        total_notional * TP_PART
    )

    runner_notional = (
        total_notional * RUNNER_PART
    )

    tp_distance = (
        TP_PCT / 100.0
    )

    liquidation_distance = max(
        1.0 / LEV - MMR,
        0.001
    )

    stop_distance = None

    if (
        SL > 0
        and SL / 100.0 < liquidation_distance
    ):

        stop_distance = SL / 100.0


    # =====================================================
    # HELPERS
    # =====================================================

    def side():

        if position == 1:
            return "LONG"

        return "SHORT"


    def pnl(nominal, exit_price):

        if position == 1:

            return (
                nominal
                * (
                    exit_price
                    / entry_price
                    - 1.0
                )
            )

        return (
            nominal
            * (
                entry_price
                / exit_price
                - 1.0
            )
        )


    def fee(nominal):

        return (
            nominal
            * FEE_PCT
            / 100.0
            * 2.0
        )


    def record(
        portion,
        exit_price,
        exit_time,
        status,
        nominal
    ):

        gross = pnl(
            nominal,
            exit_price
        )

        fees = fee(
            nominal
        )

        net = gross - fees

        if status == "LIQUIDATED":

            net = -nominal

        trades.append({

            "trade_id":
                trade_id,

            "tf":
                tf,

            "side":
                side(),

            "entry_time":
                entry_time,

            "entry_price":
                entry_price,

            "exit_time":
                exit_time,

            "exit_price":
                exit_price,

            "portion":
                portion,

            "gross_pnl":
                round(gross, 2),

            "fees":
                round(fees, 2),

            "net_pnl":
                round(net, 2),

            "status":
                status
        })


    # =====================================================
    # MAIN LOOP
    # =====================================================

    for i in range(
        2 * length,
        len(df)
    ):

        # =================================================
        # MANAGE EXISTING POSITION
        # =================================================

        if position != 0:

            # =============================================
            # BEFORE 75% TP
            # =============================================

            if not tp_hit:

                if position == 1:

                    tp_price = (
                        entry_price
                        * (
                            1
                            + tp_distance
                        )
                    )

                    tp_reached = (
                        high[i]
                        >= tp_price
                    )

                else:

                    tp_price = (
                        entry_price
                        * (
                            1
                            - tp_distance
                        )
                    )

                    tp_reached = (
                        low[i]
                        <= tp_price
                    )


                # =========================================
                # 75% TP HIT
                # =========================================

                if tp_reached:

                    record(
                        "75%",
                        tp_price,
                        times.iloc[i],
                        "TP 2%",
                        tp_notional
                    )

                    tp_hit = True


                # =========================================
                # ORIGINAL SL
                # ONLY BEFORE TP
                # =========================================

                if (
                    not tp_hit
                    and stop_distance
                ):

                    if position == 1:

                        adverse = (
                            entry_price
                            - low[i]
                        ) / entry_price

                        if adverse >= stop_distance:

                            stop_price = (
                                entry_price
                                * (
                                    1
                                    - stop_distance
                                )
                            )

                            record(
                                "100%",
                                stop_price,
                                times.iloc[i],
                                "STOP LOSS",
                                total_notional
                            )

                            position = 0
                            entry_price = None
                            entry_time = None
                            tp_hit = False

                            continue

                    else:

                        adverse = (
                            high[i]
                            - entry_price
                        ) / entry_price

                        if adverse >= stop_distance:

                            stop_price = (
                                entry_price
                                * (
                                    1
                                    + stop_distance
                                )
                            )

                            record(
                                "100%",
                                stop_price,
                                times.iloc[i],
                                "STOP LOSS",
                                total_notional
                            )

                            position = 0
                            entry_price = None
                            entry_time = None
                            tp_hit = False

                            continue


            # =============================================
            # RUNNER BREAKEVEN
            # =============================================

            if (
                position != 0
                and tp_hit
            ):

                if position == 1:

                    if low[i] <= entry_price:

                        record(
                            "25%",
                            entry_price,
                            times.iloc[i],
                            "BREAKEVEN",
                            runner_notional
                        )

                        position = 0
                        entry_price = None
                        entry_time = None
                        tp_hit = False

                        continue

                else:

                    if high[i] >= entry_price:

                        record(
                            "25%",
                            entry_price,
                            times.iloc[i],
                            "BREAKEVEN",
                            runner_notional
                        )

                        position = 0
                        entry_price = None
                        entry_time = None
                        tp_hit = False

                        continue


            # =============================================
            # LIQUIDATION
            # =============================================

            if position != 0:

                if position == 1:

                    adverse = (
                        entry_price
                        - low[i]
                    ) / entry_price

                    if adverse >= liquidation_distance:

                        liquidation_price = (
                            entry_price
                            * (
                                1
                                - liquidation_distance
                            )
                        )

                        if tp_hit:

                            record(
                                "25%",
                                liquidation_price,
                                times.iloc[i],
                                "LIQUIDATED",
                                runner_notional
                            )

                        else:

                            record(
                                "100%",
                                liquidation_price,
                                times.iloc[i],
                                "LIQUIDATED",
                                total_notional
                            )

                        position = 0
                        entry_price = None
                        entry_time = None
                        tp_hit = False

                        continue

                else:

                    adverse = (
                        high[i]
                        - entry_price
                    ) / entry_price

                    if adverse >= liquidation_distance:

                        liquidation_price = (
                            entry_price
                            * (
                                1
                                + liquidation_distance
                            )
                        )

                        if tp_hit:

                            record(
                                "25%",
                                liquidation_price,
                                times.iloc[i],
                                "LIQUIDATED",
                                runner_notional
                            )

                        else:

                            record(
                                "100%",
                                liquidation_price,
                                times.iloc[i],
                                "LIQUIDATED",
                                total_notional
                            )

                        position = 0
                        entry_price = None
                        entry_time = None
                        tp_hit = False

                        continue


        # =================================================
        # PIVOT DETECTION
        # =================================================

        pivot_index = i - length

        window_high = high[
            pivot_index - length:
            i + 1
        ]

        window_low = low[
            pivot_index - length:
            i + 1
        ]

        pivot_high = (
            high[pivot_index]
            == window_high.max()
            and
            (
                window_high
                == high[pivot_index]
            ).sum()
            == 1
        )

        pivot_low = (
            low[pivot_index]
            == window_low.min()
            and
            (
                window_low
                == low[pivot_index]
            ).sum()
            == 1
        )

        if not (
            pivot_high
            or pivot_low
        ):

            continue

        if times.iloc[i] < cutoff:

            continue

        pivots += 1

        # Pivot HIGH -> SHORT
        # Pivot LOW  -> LONG

        new_direction = (
            -1
            if pivot_high
            else 1
        )

        # Same direction = nothing
        if new_direction == position:

            continue


        # =================================================
        # OPPOSITE PIVOT
        # =================================================

        if position != 0:

            if tp_hit:

                record(
                    "25%",
                    close[i],
                    times.iloc[i],
                    "closed",
                    runner_notional
                )

            else:

                record(
                    "100%",
                    close[i],
                    times.iloc[i],
                    "closed",
                    total_notional
                )


        # =================================================
        # OPEN NEW POSITION
        # =================================================

        trade_id += 1

        position = new_direction

        entry_price = close[i]

        entry_time = times.iloc[i]

        tp_hit = False


    # =====================================================
    # CLOSE OPEN POSITION AT LAST CLOSE
    # =====================================================

    if position != 0:

        if tp_hit:

            record(
                "25%",
                close[-1],
                times.iloc[-1],
                "open (MTM)",
                runner_notional
            )

        else:

            record(
                "100%",
                close[-1],
                times.iloc[-1],
                "open (MTM)",
                total_notional
            )


    return (
        pd.DataFrame(trades),
        pivots
    )


# =========================================================
# SUMMARY
# =========================================================

def summarize(
    tf,
    trades,
    pivots,
    error=""
):

    result = {

        "tf": tf,
        "pivots": pivots,
        "trades": 0,

        "long": 0,
        "short": 0,

        "wins": 0,
        "win_pct": 0.0,

        "net_pnl": 0.0,
        "realised": 0.0,
        "open_mtm": 0.0,

        "liq": 0,
        "sl_hits": 0,
        "tp_hits": 0,
        "breakeven": 0,

        "fees": 0.0,
        "profit_factor": 0.0,
        "max_dd": 0.0,

        "note": error
    }


    if trades.empty:

        return result


    wins = trades[
        trades.net_pnl > 0
    ]
