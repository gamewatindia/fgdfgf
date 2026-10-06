"""
Pivot High/Low (LuxAlgo-style) Backtest
ALL TIMEFRAMES: 5m, 15m, 30m, 1h, 4h

Strategy:
- Confirmed pivot LOW -> LONG
- Confirmed pivot HIGH -> SHORT
- Entry = close of confirmation candle
- No look-ahead

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

OUTPUT:
- backtest_all_trades.csv
- backtest_summary.csv
- equity_all.png
"""

import argparse
import os
import time
import pandas as pd
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

MARGIN = 1000.0
LEV = 10.0

# Maintenance margin
MMR = 0.005

# 0.10 = 0.10% per side
FEE_PCT = 0.10

# Default SL OFF
SL = 0.0

# 75% position TP
TP_PCT = 2.0
TP_PART = 0.75
RUNNER_PART = 0.25

# Timeframes
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

        for quote in ("USDT", "USDC", "USD"):

            if pair.endswith(quote):

                return (
                    pair[:-len(quote)]
                    + "/"
                    + quote
                )

    return pair


# =========================================================
# FETCH OHLCV
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
            # TIME RANGE
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

            # ---------------------------------------------
            # DOWNLOAD IN CHUNKS
            # ---------------------------------------------

            safety_counter = 0

            while since < now:

                safety_counter += 1

                if safety_counter > 100:

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
                2 * warmup,
                100
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

            msg = (
                f"{exchange_name}: "
                f"{str(error)[:150]}"
            )

            print(
                f"[{tf}] {msg}"
            )

            errors.append(msg)

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
        total_notional
        * TP_PART
    )

    runner_notional = (
        total_notional
        * RUNNER_PART
    )

    tp_distance = (
        TP_PCT / 100.0
    )

    liquidation_distance = max(
        1.0 / LEV - MMR,
        0.001
    )

    stop_distance = None

    if SL > 0:

        candidate = SL / 100.0

        if candidate < liquidation_distance:

            stop_distance = candidate


    # =====================================================
    # HELPERS
    # =====================================================

    def get_side():

        return (
            "LONG"
            if position == 1
            else "SHORT"
        )


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

        net = gross - fees

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
            # 75% TP
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
            # ORIGINAL STOP LOSS
            # ONLY BEFORE TP
            # =============================================

            if (
                position != 0
                and not tp_hit
                and stop_distance is not None
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

                        record_trade(
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

                        record_trade(
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

        left_start = (
            pivot_index - length
        )

        if left_start < 0:

            continue

        window_high = high[
            left_start:
            i + 1
        ]

        window_low = low[
            left_start:
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

        # ================================================
        # PIVOT HIGH -> SHORT
        # PIVOT LOW  -> LONG
        # ================================================

        if pivot_high:

            new_direction = -1

        else:

            new_direction = 1


        # ================================================
        # SAME DIRECTION
        # ================================================

        if new_direction == position:

            continue


        # ================================================
        # OPPOSITE PIVOT
        # ================================================

        if position != 0:

            if tp_hit:

                record_trade(
                    "25%",
                    close[i],
                    times.iloc[i],
                    "closed",
                    runner_notional
                )

            else:

                record_trade(
                    "100%",
                    close[i],
                    times.iloc[i],
                    "closed",
                    total_notional
                )

            position = 0
            entry_price = None
            entry_time = None
            tp_hit = False


        # ================================================
        # OPEN NEW POSITION
        # ================================================

        trade_id += 1

        position = new_direction

        entry_price = close[i]

        entry_time = times.iloc[i]

        tp_hit = False


    # =====================================================
    # CLOSE OPEN POSITION AT END
    # =====================================================

    if position != 0:

        if tp_hit:

            record_trade(
                "25%",
                close[-1],
                times.iloc[-1],
                "open (MTM)",
                runner_notional
            )

        else:

            record_trade(
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

        "losses": 0,

        "win_pct": 0.0,

        "net_pnl": 0.0,

        "realised": 0.0,

        "open_mtm": 0.0,

        "fees": 0.0,

        "liq": 0,

        "sl_hits": 0,

        "tp_hits": 0,

        "breakeven": 0,

        "profit_factor": 0.0,

        "max_dd": 0.0,

        "note": error
    }

    if trades.empty:

        return result


    # =====================================================
    # BASIC
    # =====================================================

    result["long"] = int(
        (trades["side"] == "LONG").sum()
    )

    result["short"] = int(
        (trades["side"] == "SHORT").sum()
    )

    result["fees"] = round(
        trades["fees"].sum(),
        2
    )

    result["liq"] = int(
        (
            trades["status"]
            == "LIQUIDATED"
        ).sum()
    )

    result["sl_hits"] = int(
        (
            trades["status"]
            == "STOP LOSS"
        ).sum()
    )

    result["tp_hits"] = int(
        (
            trades["status"]
            == "TP 2%"
        ).sum()
    )

    result["breakeven"] = int(
        (
            trades["status"]
            == "BREAKEVEN"
        ).sum()
    )


    # =====================================================
    # GROUP PARTIALS INTO REAL TRADES
    # =====================================================

    grouped = (
        trades
        .groupby("trade_id", as_index=False)
        .agg({

            "side": "first",

            "net_pnl": "sum",

            "gross_pnl": "sum",

            "fees": "sum",

            "status": lambda x:
                "|".join(
                    x.astype(str)
                )
        })
    )

    result["trades"] = int(
        len(grouped)
    )


    # =====================================================
    # WIN / LOSS
    # =====================================================

    wins = grouped[
        grouped["net_pnl"] > 0
    ]

    losses = grouped[
        grouped["net_pnl"] < 0
    ]

    result["wins"] = int(
        len(wins)
    )

    result["losses"] = int(
        len(losses)
    )

    if result["trades"] > 0:

        result["win_pct"] = round(
            result["wins"]
            / result["trades"]
            * 100.0,
            2
        )


    # =====================================================
    # PNL
    # =====================================================

    total_pnl = grouped[
        "net_pnl"
    ].sum()

    result["net_pnl"] = round(
        total_pnl,
        2
    )


    open_rows = trades[
        trades["status"]
        == "open (MTM)"
    ]

    result["open_mtm"] = round(
        open_rows["net_pnl"].sum()
        if not open_rows.empty
        else 0.0,
        2
    )

    result["realised"] = round(
        total_pnl
        - result["open_mtm"],
        2
    )


    # =====================================================
    # PROFIT FACTOR
    # =====================================================

    gross_profit = wins[
        "net_pnl"
    ].sum()

    gross_loss = abs(
        losses[
            "net_pnl"
        ].sum()
    )

    if gross_loss > 0:

        result["profit_factor"] = round(
            gross_profit
            / gross_loss,
            3
        )

    elif gross_profit > 0:

        result["profit_factor"] = float(
            "inf"
        )


    # =====================================================
    # MAX DRAWDOWN
    # =====================================================

    equity = grouped[
        "net_pnl"
    ].cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    if len(drawdown):

        result["max_dd"] = round(
            drawdown.min(),
            2
        )


    return result


# =========================================================
# EQUITY CURVE
# =========================================================

def make_equity_chart(
    all_trades
):

    try:

        import matplotlib

        matplotlib.use(
            "Agg"
        )

        import matplotlib.pyplot as plt

        plt.figure(
            figsize=(14, 8)
        )

        plotted = False

        for tf in TFS:

            tf_trades = all_trades[
                all_trades["tf"] == tf
            ].copy()

            if tf_trades.empty:

                continue

            # Aggregate partials
            equity_data = (
                tf_trades
                .groupby(
                    "trade_id",
                    as_index=False
                )["net_pnl"]
                .sum()
            )

            equity = (
                equity_data["net_pnl"]
                .cumsum()
            )

            plt.plot(
                range(
                    1,
                    len(equity) + 1
                ),
                equity,
                label=tf
            )

            plotted = True

        plt.axhline(
            0,
            linewidth=1
        )

        plt.title(
            "Pivot Backtest Equity Curve"
        )

        plt.xlabel(
            "Completed Trades"
        )

        plt.ylabel(
            "Cumulative Net PnL (Rs)"
        )

        if plotted:

            plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            "equity_all.png",
            dpi=150
        )

        plt.close()

    except Exception as error:

        print(
            f"Equity chart warning: {error}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    global MARGIN
    global LEV
    global SL

    parser = argparse.ArgumentParser(
        description=(
            "Pivot High/Low "
            "LuxAlgo-style Backtest"
        )
    )

    parser.add_argument(
        "--pair",
        required=True,
        help="USDT pair e.g. BTCUSDT"
    )

    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Historical days"
    )

    parser.add_argument(
        "--price",
        choices=[
            "LAST_PRICE",
            "MARK_PRICE",
            "INDEX_PRICE"
        ],
        default="LAST_PRICE"
    )

    parser.add_argument(
        "--lev",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=1000.0
    )

    parser.add_argument(
        "--sl",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--first",
        default=""
    )

    parser.add_argument(
        "--length",
        type=int,
        default=50
    )

    args = parser.parse_args()


    # =====================================================
    # APPLY SETTINGS
    # =====================================================

    MARGIN = float(
        args.margin
    )

    LEV = float(
        args.lev
    )

    SL = float(
        args.sl
    )

    pair = norm_pair(
        args.pair
    )

    days = int(
        args.days
    )

    length = int(
        args.length
    )

    if days <= 0:

        raise ValueError(
            "days must be > 0"
        )

    if length < 2:

        raise ValueError(
            "length must be >= 2"
        )

    if LEV <= 0:

        raise ValueError(
            "leverage must be > 0"
        )

    if MARGIN <= 0:

        raise ValueError(
            "margin must be > 0"
        )


    # =====================================================
    # HEADER
    # =====================================================

    print()
    print(
        "=" * 80
    )

    print(
        "PIVOT HIGH/LOW BACKTEST"
    )

    print(
        "=" * 80
    )

    print(
        f"Pair       : {pair}"
    )

    print(
        f"Price      : {args.price}"
    )

    print(
        f"Days       : {days}"
    )

    print(
        f"Length     : {length}"
    )

    print(
        f"Margin     : Rs {MARGIN:.2f}"
    )

    print(
        f"Leverage   : {LEV:.1f}x"
    )

    print(
        f"Position   : Rs {MARGIN * LEV:.2f}"
    )

    print(
        f"75% TP     : {TP_PCT:.2f}%"
    )

    print(
        f"Runner     : {RUNNER_PART * 100:.0f}%"
    )

    print(
        f"SL         : "
        f"{SL:.2f}%"
    )

    print(
        f"Fee/side   : "
        f"{FEE_PCT:.2f}%"
    )

    print(
        "=" * 80
    )

    print()


    # =====================================================
    # RESULTS
    # =====================================================

    all_trade_frames = []

    summaries = []


    # =====================================================
    # RUN ALL TIMEFRAMES
    # =====================================================

    for tf in TFS:

        print()
        print(
            "=" * 80
        )

        print(
            f"RUNNING {tf}"
        )

        print(
            "=" * 80
        )

        try:

            # ---------------------------------------------
            # Fetch data
            # ---------------------------------------------

            warmup = (
                length * 2
                + 10
            )

            df = fetch(
                pair=pair,
                tf=tf,
                days=days,
                warmup=warmup,
                price=args.price,
                first=args.first
            )

            # ---------------------------------------------
            # Backtest
            # ---------------------------------------------

            trades, pivots = backtest(
                df=df,
                length=length,
                days=days,
                tf=tf
            )

            # ---------------------------------------------
            # Summary
            # ---------------------------------------------

            summary = summarize(
                tf=tf,
                trades=trades,
                pivots=pivots
            )

            summaries.append(
                summary
            )

            if not trades.empty:

                all_trade_frames.append(
                    trades
                )

            print()
            print(
                f"[{tf}] "
                f"pivots={pivots} "
                f"trades={summary['trades']} "
                f"wins={summary['wins']} "
                f"win%={summary['win_pct']:.2f} "
                f"net PnL=Rs {summary['net_pnl']:.2f}"
            )

        except Exception as error:

            print()
            print(
                f"[{tf}] ERROR:"
            )

            print(
                str(error)
            )

            summaries.append(
                summarize(
                    tf=tf,
                    trades=pd.DataFrame(),
                    pivots=0,
                    error=str(error)
                )
            )


    # =====================================================
    # COMBINE TRADES
    # =====================================================

    if all_trade_frames:

        all_trades = pd.concat(
            all_trade_frames,
            ignore_index=True
        )

    else:

        all_trades = pd.DataFrame(
            columns=[
                "trade_id",
                "tf",
                "side",
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "portion",
                "gross_pnl",
                "fees",
                "net_pnl",
                "status"
            ]
        )


    # =====================================================
    # SAVE ALL TRADES
    # =====================================================

    all_trades.to_csv(
        "backtest_all_trades.csv",
        index=False
    )


    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        "backtest_summary.csv",
        index=False
    )


    # =====================================================
    # EQUITY CHART
    # =====================================================

    if not all_trades.empty:

        make_equity_chart(
            all_trades
        )

    else:

        # Create an empty valid PNG
        try:

            import matplotlib

            matplotlib.use(
                "Agg"
            )

            import matplotlib.pyplot as plt

            plt.figure(
                figsize=(14, 8)
            )

            plt.title(
                "Pivot Backtest - No Trades"
            )

            plt.axhline(
                0
            )

            plt.tight_layout()

            plt.savefig(
                "equity_all.png",
                dpi=150
            )

            plt.close()

        except Exception as error:

            print(
                f"Could not create "
                f"empty equity chart: {error}"
            )


    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    print()
    print(
        "=" * 100
    )

    print(
        "BACKTEST SUMMARY"
    )

    print(
        "=" * 100
    )

    if not summary_df.empty:

        display_columns = [
            "tf",
            "pivots",
            "trades",
            "long",
            "short",
            "wins",
            "losses",
            "win_pct",
            "net_pnl",
            "realised",
            "open_mtm",
            "fees",
            "liq",
            "sl_hits",
            "tp_hits",
            "breakeven",
            "profit_factor",
            "max_dd"
        ]

        print(
            summary_df[
                display_columns
            ].to_string(
                index=False
            )
        )


    # =====================================================
    # FINAL FILE CHECK
    # =====================================================

    print()
    print(
        "=" * 100
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 100
    )

    for filename in [
        "backtest_all_trades.csv",
        "backtest_summary.csv",
        "equity_all.png"
    ]:

        if os.path.exists(filename):

            size = os.path.getsize(
                filename
            )

            print(
                f"OK   {filename} "
                f"({size} bytes)"
            )

        else:

            print(
                f"ERROR {filename} MISSING"
            )


    print()
    print(
        "BACKTEST COMPLETED SUCCESSFULLY"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()