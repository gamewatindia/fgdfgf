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
                f"[{tf}] data: