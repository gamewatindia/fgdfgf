"""
Pivot High/Low Backtest
ALL TIMEFRAMES:
5m, 15m, 30m, 1h, 4h

STRATEGY
--------
Confirmed Pivot LOW  -> LONG
Confirmed Pivot HIGH -> SHORT

ENTRY
-----
Entry = close of confirmation candle.

POSITION
--------
Margin       = user input
Leverage     = user input
Notional     = Margin x Leverage

75% position:
    TP = +2% price movement

25% position:
    Runner

After 75% TP:
    Runner SL = BREAKEVEN

Runner exits on:
    - Breakeven
    - Opposite pivot
    - Liquidation

Before TP:
    - Optional SL
    - Opposite pivot = close full + reverse
    - Liquidation

Outputs:
    backtest_all_trades.csv
    backtest_summary.csv
    equity_all.png
"""

import argparse
import os
import time

import numpy as np
import pandas as pd


# =========================================================
# DEFAULT SETTINGS
# =========================================================

DEFAULT_MARGIN = 1000.0
DEFAULT_LEV = 10.0

MMR = 0.005

# 0.10% per side
FEE_PCT = 0.10

TP_PCT = 2.0

TP_PART = 0.75
RUNNER_PART = 0.25

TFS = [
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
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
        "bybit",
    ],

    "MARK_PRICE": [
        "gateio",
        "bitget",
        "okx",
        "bybit",
        "binanceusdm",
        "kucoinfutures",
    ],

    "INDEX_PRICE": [
        "gateio",
        "bitget",
        "okx",
        "bybit",
        "binanceusdm",
    ],
}


# =========================================================
# NORMALIZE PAIR
# =========================================================

def norm_pair(pair):

    pair = pair.upper().strip()

    pair = pair.replace("-", "/")
    pair = pair.replace("_", "/")

    if "/" not in pair:

        for quote in ["USDT", "USDC", "USD"]:

            if pair.endswith(quote):

                base = pair[:-len(quote)]

                return f"{base}/{quote}"

    return pair


# =========================================================
# FETCH DATA
# =========================================================

def fetch_data(
    pair,
    tf,
    days,
    warmup,
    price_type,
    first_exchange="",
):

    import ccxt

    errors = []

    exchanges = []

    if first_exchange:
        exchanges.append(first_exchange)

    for ex in EXCHANGES[price_type]:

        if ex not in exchanges:
            exchanges.append(ex)

    for name in exchanges:

        try:

            if not hasattr(ccxt, name):
                raise ValueError(
                    f"CCXT exchange not available: {name}"
                )

            ex = getattr(ccxt, name)({
                "enableRateLimit": True
            })

            ex.load_markets()

            # ---------------------------------------------
            # SYMBOL
            # ---------------------------------------------

            if price_type == "LAST_PRICE":

                symbol = pair

            else:

                # Futures symbol normally:
                # BTC/USDT:USDT
                symbol = (
                    pair
                    + ":"
                    + pair.split("/")[1]
                )

            # Some exchanges may not have exact futures
            # symbol. Try normal symbol as fallback.
            if symbol not in ex.markets:

                if pair in ex.markets:

                    symbol = pair

                else:

                    raise ValueError(
                        f"{symbol} not listed"
                    )

            # ---------------------------------------------
            # FUNCTION
            # ---------------------------------------------

            if price_type == "LAST_PRICE":

                if not ex.has.get("fetchOHLCV"):
                    raise ValueError(
                        "fetchOHLCV not supported"
                    )

                fn = ex.fetch_ohlcv

            elif price_type == "MARK_PRICE":

                if not ex.has.get("fetchMarkOHLCV"):
                    raise ValueError(
                        "mark OHLCV not supported"
                    )

                fn = ex.fetch_mark_ohlcv

            else:

                if not ex.has.get("fetchIndexOHLCV"):
                    raise ValueError(
                        "index OHLCV not supported"
                    )

                fn = ex.fetch_index_ohlcv

            # ---------------------------------------------
            # TIME
            # ---------------------------------------------

            tf_ms = (
                ex.parse_timeframe(tf)
                * 1000
            )

            now = ex.milliseconds()

            since = (
                now
                - days * 86400000
                - warmup * tf_ms
            )

            rows = []

            # ---------------------------------------------
            # FETCH IN CHUNKS
            # ---------------------------------------------

            while since < now:

                candles = fn(
                    symbol,
                    tf,
                    since=since,
                    limit=1000,
                )

                if not candles:
                    break

                rows.extend(candles)

                next_since = (
                    candles[-1][0]
                    + tf_ms
                )

                if next_since <= since:
                    break

                since = next_since

                time.sleep(
                    max(
                        ex.rateLimit / 1000,
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
                rows,
                columns=[
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    *[
                        f"x{i}"
                        for i in range(
                            max(
                                0,
                                len(rows[0]) - 6
                            )
                        )
                    ],
                ],
            )

            df = df[
                [
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            ]

            df["time"] = pd.to_datetime(
                df["time"],
                unit="ms",
                utc=True,
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
            ]:

                df