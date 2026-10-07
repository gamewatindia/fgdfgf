import argparse
import time
import os
import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================

MARGIN_DEFAULT = 1000.0
LEV_DEFAULT = 10.0

MMR = 0.005
FEE_PCT = 0.10

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

    p = pair.upper()

    p = p.replace("-", "/")
    p = p.replace("_", "/")

    if "/" not in p:

        for q in ("USDT", "USDC", "USD"):

            if p.endswith(q):

                return (
                    p[:-len(q)]
                    + "/"
                    + q
                )

    return p


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

    order = (
        [first]
        if first
        else []
    )

    order += [
        x
        for x in EXCHANGES[price]
        if x != first
    ]

    errors = []

    for name in order:

        exchange = None

        try:

            exchange = getattr(
                ccxt,
                name
            )({
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

                fn = exchange.fetch_ohlcv

            elif price == "MARK_PRICE":

                if not exchange.has.get(
                    "fetchMarkOHLCV"
                ):

                    raise ValueError(
                        "mark OHLCV unsupported"
                    )

                fn = exchange.fetch_mark_ohlcv

            else:

                if not exchange.has.get(
                    "fetchIndexOHLCV"
                ):

                    raise ValueError(
                        "index OHLCV unsupported"
                    )

                fn = exchange.fetch_index_ohlcv

            # ---------------------------------------------
            # TIME
            # ---------------------------------------------

            tf_ms = (
                exchange.parse_timeframe(tf)
                * 1000
            )

            now = exchange.milliseconds()

            since = (
                now
                - days * 86400000
                - warmup * tf_ms
            )

            rows = []

            # ---------------------------------------------
            # DOWNLOAD
            # ---------------------------------------------

            while since < now:

                batch = fn(
                    symbol,
                    tf,
                    since=since,
                    limit=1000
                )

                if not batch:

                    break

                rows.extend(batch)

                next_since = (
                    batch[-1][0]
                    + tf_ms
                )

                if next_since <= since:

                    break

                since = next_since

                time.sleep(
                    max(
                        exchange.rateLimit,
                        50
                    ) / 1000.0
                )

            if len(rows) < max(
                2 * warmup,
                100
            ):

                raise ValueError(
                    f"only {len(rows)} candles"
                )

            # ---------------------------------------------
            # DATAFRAME
            # ---------------------------------------------

            df = (
                pd.DataFrame(rows)
                .iloc[:, :5]
            )

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
                f"{name} "
                f"{symbol} "
                f"{price} "
                f"({len(df)} candles)"
            )

            return df

        except Exception as e:

            errors.append(
                f"{name}: "
                f"{str(e)[:120]}"
            )

        finally:

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
    tf,
    margin,
    lev
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
        - pd.Timedelta(days=days)
    )

    trades = []

    position = 0

    entry_price = None
    entry_time = None

    sl_price = None

    trade_id = 0

    tp_hit = False

    # =====================================================
    # POSITION SIZE
    # =====================================================

    total_notional = (
        margin * lev
    )

    tp_notional = (
        total_notional
        * TP_PART
    )

    runner_notional = (
        total_notional
        * RUNNER_PART
    )

    # =====================================================
    # LIQUIDATION
    # =====================================================

    liquidation_distance = max(
        1.0 / lev - MMR,
        0.001
    )


    # =====================================================
    # HELPERS
    # =====================================================

    def side():

        if position == 1:

            return "LONG"

        return "SHORT"


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


    def calculate_fee(
        nominal
    ):

        return (
            nominal
            * FEE_PCT
            / 100.0
            * 2.0
        )


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
            gross
            - fees
        )

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
                round(
                    entry_price,
                    8
                ),

            "entry_sl":
                round(
                    sl_price,
                    8
                ),

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


    def reset_position():

        nonlocal position
        nonlocal entry_price
        nonlocal entry_time
        nonlocal sl_price
        nonlocal tp_hit

        position = 0

        entry_price = None

        entry_time = None

        sl_price = None

        tp_hit = False


    # =====================================================
    # MAIN LOOP
    # =====================================================

    for i in range(
        max(
            2 * length,
            length + 2
        ),
        len(df)
    ):

        # =================================================
        # MANAGE EXISTING POSITION
        # =================================================

        if position != 0:

            # =============================================
            # BEFORE TP
            # =============================================

            if not tp_hit:

                # -----------------------------------------
                # ENTRY CANDLE SL
                # -----------------------------------------

                if position == 1:

                    sl_hit = (
                        low[i]
                        <= sl_price
                    )

                else:

                    sl_hit = (
                        high[i]
                        >= sl_price
                    )

                # -----------------------------------------
                # TP PRICE
                # -----------------------------------------

                if position == 1:

                    tp_price = (
                        entry_price
                        * (
                            1
                            + TP_PCT / 100
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
                            - TP_PCT / 100
                        )
                    )

                    tp_reached = (
                        low[i]
                        <= tp_price
                    )

                # -----------------------------------------
                # SL FIRST
                # -----------------------------------------

                if sl_hit:

                    record_trade(
                        "100%",
                        sl_price,
                        times.iloc[i],
                        "ENTRY CANDLE SL",
                        total_notional
                    )

                    reset_position()

                    continue

                # -----------------------------------------
                # 75% TP
                # -----------------------------------------

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
            # RUNNER AFTER TP
            # =============================================

            if (
                position != 0
                and tp_hit
            ):

                # Runner SL = BREAKEVEN

                if position == 1:

                    breakeven_hit = (
                        low[i]
                        <= entry_price
                    )

                else:

                    breakeven_hit = (
                        high[i]
                        >= entry_price
                    )

                if breakeven_hit:

                    record_trade(
                        "25%",
                        entry_price,
                        times.iloc[i],
                        "BREAKEVEN",
                        runner_notional
                    )

                    reset_position()

                    continue


            # =============================================
            # LIQUIDATION
            # =============================================

            if position != 0:

                if position == 1:

                    liquidation_hit = (
                        low[i]
                        <= (
                            entry_price
                            * (
                                1
                                - liquidation_distance
                            )
                        )
                    )

                else:

                    liquidation_hit = (
                        high[i]
                        >= (
                            entry_price
                            * (
                                1
                                + liquidation_distance
                            )
                        )
                    )

                if liquidation_hit:

                    if position == 1:

                        liquidation_price = (
                            entry_price
                            * (
                                1
                                - liquidation_distance
                            )
                        )

                    else:

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
                            total_notional
                        )

                    reset_position()

                    continue


        # =================================================
        # PIVOT DETECTION
        # =================================================

        pivot_index = (
            i - length
        )

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

        if (
            times.iloc[i]
            < cutoff
        ):

            continue

        # =================================================
        # D