"""
PIVOT HIGH/LOW LUXALGO-STYLE BACKTEST

TIMEFRAMES:
5m, 15m, 30m, 1h, 4h

ENTRY:
Confirmed Pivot LOW  -> LONG
Confirmed Pivot HIGH -> SHORT

ENTRY PRICE:
Confirmation candle CLOSE

INITIAL STOP LOSS:
LONG  -> Confirmation candle LOW
SHORT -> Confirmation candle HIGH

POSITION:
Margin    = Rs 1000
Leverage  = 10x
Notional  = Rs 10000

TAKE PROFIT:
75% position closes at +2% price movement.

RUNNER:
25% remains after TP.

After 75% TP:
Runner SL = Entry price (BREAKEVEN)

Runner closes at:
- Breakeven
- Opposite pivot
- Liquidation
- End of data

OUTPUT:
backtest_all_trades.csv
backtest_summary.csv
equity_all.png
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

# Fee per side = 0.10%
FEE_PCT = 0.10

# 75% position TP
TP_PCT = 2.0

TP_PART = 0.75
RUNNER_PART = 0.25


# =========================================================
# TIMEFRAMES
# =========================================================

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

    pair = pair.upper().strip()

    pair = pair.replace("-", "/")
    pair = pair.replace("_", "/")

    if "/" not in pair:

        for quote in (
            "USDT",
            "USDC",
            "USD"
        ):

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

def fetch(
    pair,
    tf,
    days,
    warmup,
    price,
    first=""
):

    import ccxt

    errors = []

    order = []

    if first:
        order.append(first)

    for exchange_name in EXCHANGES[price]:

        if exchange_name not in order:

            order.append(exchange_name)

    for exchange_name in order:

        exchange = None

        try:

            print(
                f"[{tf}] Trying {exchange_name}..."
            )

            exchange_class = getattr(
                ccxt,
                exchange_name
            )

            exchange = exchange_class({
                "enableRateLimit": True
            })

            exchange.load_markets()

            # ---------------------------------------------
            # SYMBOL
            # ---------------------------------------------

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

            # ---------------------------------------------
            # FETCH FUNCTION
            # ---------------------------------------------

            if price == "LAST_PRICE":

                if not exchange.has.get(
                    "fetchOHLCV"
                ):

                    raise ValueError(
                        "fetchOHLCV not supported"
                    )

                fetch_function = (
                    exchange.fetch_ohlcv
                )

            elif price == "MARK_PRICE":

                if not exchange.has.get(
                    "fetchMarkOHLCV"
                ):

                    raise ValueError(
                        "mark OHLCV not supported"
                    )

                fetch_function = (
                    exchange.fetch_mark_ohlcv
                )

            else:

                if not exchange.has.get(
                    "fetchIndexOHLCV"
                ):

                    raise ValueError(
                        "index OHLCV not supported"
                    )

                fetch_function = (
                    exchange.fetch_index_ohlcv
                )

            # ---------------------------------------------
            # TIME
            # ---------------------------------------------

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

            safety = 0

            while since < now:

                safety += 1

                if safety > 100:

                    break

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
                    max(
                        exchange.rateLimit / 1000,
                        0.05
                    )
                )

            if len(rows) < max(
                100,
                warmup * 2
            ):

                raise ValueError(
                    f"only {len(rows)} candles"
                )

            # ---------------------------------------------
            # DATAFRAME
            # ---------------------------------------------

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

            for col in [
                "open",
                "high",
                "low",
                "close"
            ]:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df = (
                df
                .dropna()
                .drop_duplicates(
                    "time"
                )
                .sort_values(
                    "time"
                )
                .reset_index(
                    drop=True
                )
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

            message = (
                f"{exchange_name}: "
                f"{str(error)[:150]}"
            )

            print(
                f"[{tf}] {message}"
            )

            errors.append(
                message
            )

            try:

                if exchange is not None:

                    exchange.close()

            except Exception:

                pass

    raise RuntimeError(
        " | ".join(errors)
    )


# =========================================================
# BACKTEST
# =========================================================

def backtest(
    df,
    length,
    days,
    tf
):

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
        - pd.Timedelta(
            days=days
        )
    )

    trades = []

    pivots = 0

    # -----------------------------------------------------
    # POSITION
    # -----------------------------------------------------

    position = 0

    # 1 = LONG
    # -1 = SHORT

    entry_price = None
    entry_time = None

    # NEW:
    # Entry candle based SL
    entry_sl = None

    trade_id = 0

    tp_hit = False

    # -----------------------------------------------------
    # POSITION SIZE
    # -----------------------------------------------------

    total_notional = (
        MARGIN * LEV
    )

    tp_notional = (
        total_notional
        * TP_PART
    )

    runner_notional = (
        total_notional
        * RUNNER_PART
    )

    # -----------------------------------------------------
    # TP
    # -----------------------------------------------------

    tp_distance = (
        TP_PCT / 100.0
    )

    # -----------------------------------------------------
    # LIQUIDATION
    # -----------------------------------------------------

    liquidation_distance = max(
        1.0 / LEV - MMR,
        0.001
    )


    # =====================================================
    # HELPER: SIDE
    # =====================================================

    def get_side():

        if position == 1:

            return "LONG"

        return "SHORT"


    # =====================================================
    # HELPER: PNL
    # =====================================================

    def calculate_pnl(
        nominal,
        exit_price
    ):

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


    # =====================================================
    # HELPER: FEES
    # =====================================================

    def calculate_fee(
        nominal
    ):

        return (
            nominal
            * FEE_PCT
            / 100.0
            * 2.0
        )


    # =====================================================
    # RECORD TRADE
    # =====================================================

    def record_trade(
        portion,
        exit_price,
        exit_time,
        status,
        nominal
    ):

        gross = calculate_pnl(
            nominal,
            exit_price
        )

        fees = calculate_fee(
            nominal
        )

        net = (
            gross - fees
        )

        # Liquidation
        if status == "LIQUIDATED":

            net = -nominal

        trades.append({

            "trade_id":
                trade_id,

            "tf":
                tf,

            "side":
                get_side(),

            "entry_time":
                entry_time,

            "entry_price":
                round(
                    entry_price,
                    8
                ),

            "entry_sl":
                round(
                    entry_sl,
                    8
                )
                if entry_sl is not None
                else None,

            "exit_time":
                exit_time,

            "exit_price":
                round(
                    exit_price,
                    8
                ),

            "portion":
                portion,

            "gross_pnl":
                round(
                    gross,
                    2
                ),

            "fees":
                round(
                    fees,
                    2
                ),

            "net_pnl":
                round(
                    net,
                    2
                ),

            "status":
                status
        })


    # =====================================================
    # MAIN LOOP
    # =====================================================

    start_index = max(
        2 * length,
        length + 2
    )

    for i in range(
        start_index,
        len(df)
    ):

        # =================================================
        # MANAGE EXISTING POSITION
        # =================================================

        if position != 0:

            # =============================================
            # 1. 75% TP
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

                if tp_reached:

                    record_trade(
                        "75%",
                        tp_price,
                        times.iloc[i],
                        "TP 2%",
                        tp_notional
                    )

                    tp_hit = True


            # =============================================
            # 2. ENTRY CANDLE SL
            #
            # LONG  -> entry candle LOW
            # SHORT -> entry candle HIGH
            #
            # ONLY BEFORE TP
            # =============================================

            if (
                position != 0
                and not tp_hit
                and entry_sl is not None
            ):

                if position == 1:

                    # LONG:
                    # candle low touches SL

                    if low[i] <= entry_sl:

                        record_trade(
                            "100%",
                            entry_sl,
                            times.iloc[i],
                            "ENTRY CANDLE SL",
                            total_notional
                        )

                        position = 0

                        entry_price = None

                        entry_time = None

                        entry_sl = None

                        tp_hit = False

                        continue

                else:

                    # SHORT:
                    # candle high touches SL

                    if high[i] >= entry_sl:

                        record_trade(
                            "100%",
                            entry_sl,
                            times.iloc[i],
                            "ENTRY CANDLE SL",
                            total_notional
                        )

                        position = 0

                        entry_price = None

                        entry_time = None

                        entry_sl = None

                        tp_hit = False

                        continue


            # =============================================
            # 3. RUNNER BREAKEVEN
            # =============================================

            if (
                position != 0
                and tp_hit
            ):

                if position == 1:

                    if low[i] <= entry_price:

                        record_trade(
                            "25%",
                            entry_price,
                            times.iloc[i],
                            "BREAKEVEN",
                            runner_notional
                        )

                        position = 0

                        entry_price = None

                        entry_time = None

                        entry_sl = None

                        tp_hit = False

                        continue

                else:

                    if high[i] >= entry_price:

                        record_trade(
                            "25%",
                            entry_price,
                            times.iloc[i],
                            "BREAKEVEN",
                            runner_notional
                        )

                        position = 0

                        entry_price = None

                        entry_time = None

                        entry_sl = None

                        tp_hit = False

                        continue


            # =============================================
            # 4. LIQUIDATION
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

                            record_trade(
                                "25%",
                                liquidation_price,
                                times.iloc[i],
                                "LIQUIDATED",
                                runner_notional
                            )

                        else:

                            record_trade(
                                "100%",
                                liquidation_price,
                                times.iloc[i],
                                "LIQUIDATED",
                                total_notional
                            )

                        position = 0

                        entry_price = None

                        entry_time = None

                        entry_sl = None

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

                            record_trade(
                                "25%",
                                liquidation_price,
                                times.iloc[i],
                                "LIQUIDATED",
                                runner_notional
                            )

                        else:

                            record_trade(
                                "100%",
                                liquidation_price,
                                times.iloc[i],
                                "LIQUIDATED",
                                total