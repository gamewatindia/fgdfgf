"""
Pivot High/Low (LuxAlgo-style) Backtest - ALL TIMEFRAMES in ONE run

Stop-and-reverse:
confirmed pivot LOW -> LONG
confirmed pivot HIGH -> SHORT

ENTRY:
Entry = close of the bar where pivot is confirmed.
No look-ahead.

POSITION MANAGEMENT:
-----------------------------------------
Total margin        = Rs 1000
Leverage             = 10x
Total position       = Rs 10,000

75% position:
    Fixed TP = +2% price move

25% position:
    Runner
    After 75% TP is hit -> SL moves to BREAKEVEN
    Breakeven starts from the NEXT candle after TP.
    Runner can continue until:
        - Breakeven
        - Opposite pivot/reversal
        - Liquidation

OPTIONAL ORIGINAL SL:
    --sl 4 means 4% price movement against position.
    This applies only BEFORE the 2% TP is hit.

Outputs:
    backtest_all_trades.csv
    backtest_summary.csv
    equity_all.png
"""

import argparse
import os
import time
import pandas as pd


# =========================================================
# GLOBAL CONFIGURATION
# =========================================================

MARGIN = 1000.0
LEV = 10.0

MMR = 0.005

FEE_PCT = 0.10

SL = 0.0


# =========================================================
# PARTIAL TP CONFIGURATION
# =========================================================

TP_PCT = 2.0

TP_PART = 0.75
RUNNER_PART = 0.25

BREAKEVEN_AFTER_TP = True


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
    ],
}


# =========================================================
# NORMALIZE PAIR
# =========================================================

def norm_pair(p):

    p = p.upper()
    p = p.replace("-", "/")
    p = p.replace("_", "/")

    if "/" not in p:

        for q in ("USDT", "USDC", "USD"):

            if p.endswith(q):

                return (
                    p[: -len(q)]
                    + "/"
                    + q
                )

    return p


# =========================================================
# FETCH OHLCV DATA
# =========================================================

def fetch(pair, tf, days, warmup, price, first=""):

    import ccxt

    errs = []

    order = (
        [first] if first else []
    ) + [
        e
        for e in EXCHANGES[price]
        if e != first
    ]

    for name in order:

        try:

            ex = getattr(
                ccxt,
                name
            )(
                {
                    "enableRateLimit": True
                }
            )

            ex.load_markets()

            if price == "LAST_PRICE":

                sym = pair

            else:

                sym = (
                    pair
                    + ":"
                    + pair.split("/")[1]
                )

            if sym not in ex.markets:

                raise ValueError(
                    f"{sym} not listed"
                )

            fn = {

                "LAST_PRICE":
                    ex.fetch_ohlcv,

                "MARK_PRICE":
                    ex.fetch_mark_ohlcv,

                "INDEX_PRICE":
                    ex.fetch_index_ohlcv

            }[price]

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

            while since < now:

                b = fn(
                    sym,
                    tf,
                    since=since,
                    limit=1000
                )

                if not b:
                    break

                rows += b

                nxt = (
                    b[-1][0]
                    + tf_ms
                )

                if nxt <= since:
                    break

                since = nxt

                time.sleep(
                    ex.rateLimit / 1000
                )

            if len(rows) < 2 * warmup:

                raise ValueError(
                    f"only {len(rows)} candles"
                )

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
                f"{sym} "
                f"{price} "
                f"({len(df)} candles)"
            )

            return df

        except Exception as e:

            errs.append(
                f"{name}: "
                f"{str(e)[:100]}"
            )

    raise RuntimeError(
        " | ".join(errs)
    )


# =========================================================
# BACKTEST
# =========================================================

def backtest(df, length, days, tf):

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    t = df["time"]

    cutoff = (
        t.iloc[-1]
        - pd.Timedelta(days=days)
    )

    trades = []

    npv = 0

    # =====================================================
    # POSITION STATE
    # =====================================================

    pos = 0

    epx = None
    et = None

    trade_id = 0

    # =====================================================
    # POSITION SIZE
    # =====================================================

    notional = MARGIN * LEV

    tp_notional = (
        notional * TP_PART
    )

    runner_notional = (
        notional * RUNNER_PART
    )

    # =====================================================
    # TP
    # =====================================================

    tp_distance = TP_PCT / 100.0

    # =====================================================
    # LIQUIDATION
    # =====================================================

    liq = max(
        1 / LEV - MMR,
        0.001
    )

    # =====================================================
    # ORIGINAL SL
    # =====================================================

    sd = (
        SL / 100.0
        if 0 < SL / 100.0 < liq
        else None
    )

    # =====================================================
    # TP STATE
    # =====================================================

    tp_hit = False

    # VERY IMPORTANT:
    # BE starts ONLY from candle after TP candle.
    tp_bar_index = None


    # =====================================================
    # CLOSE 75% TP
    # =====================================================

    def close_tp(px, tm, bar_index):

        nonlocal tp_hit
        nonlocal tp_bar_index

        if pos == 1:

            gross = (
                tp_notional
                * (px / epx - 1)
            )

        else:

            gross = (
                tp_notional
                * (epx / px - 1)
            )

        fee = (
            tp_notional
            * FEE_PCT
            / 100
            * 2
        )

        net = gross - fee

        trades.append({

            "trade_id": trade_id,

            "tf": tf,

            "side":
                "LONG"
                if pos == 1
                else "SHORT",

            "entry_time": et,

            "entry_price": epx,

            "exit_time": tm,

            "exit_price": px,

            "portion": "75%",

            "gross_pnl": round(
                gross,
                2
            ),

            "fees": round(
                fee,
                2
            ),

            "net_pnl": round(
                net,
                2
            ),

            "status": "TP 2%"

        })

        tp_hit = True
        tp_bar_index = bar_index


    # =====================================================
    # CLOSE 25% RUNNER
    # =====================================================

    def close_runner(
        px,
        tm,
        status
    ):

        if pos == 1:

            gross = (
                runner_notional
                * (px / epx - 1)
            )

        else:

            gross = (
                runner_notional
                * (epx / px - 1)
            )

        fee = (
            runner_notional
            * FEE_PCT
            / 100
            * 2
        )

        net = gross - fee

        if status == "LIQUIDATED":

            net = -runner_notional

        trades.append({

            "trade_id": trade_id,

            "tf": tf,

            "side":
                "LONG"
                if pos == 1
                else "SHORT",

            "entry_time": et,

            "entry_price": epx,

            "exit_time": tm,

            "exit_price": px,

            "portion": "25%",

            "gross_pnl": round(
                gross,
                2
            ),

            "fees": round(
                fee,
                2
            ),

            "net_pnl": round(
                net,
                2
            ),

            "status": status

        })


    # =====================================================
    # CLOSE FULL POSITION
    # =====================================================

    def close_full(
        px,
        tm,
        status
    ):

        if pos == 1:

            gross = (
                notional
                * (px / epx - 1)
            )

        else:

            gross = (
                notional
                * (epx / px - 1)
            )

        fee = (
            notional
            * FEE_PCT
            / 100
            * 2
        )

        net = max(
            gross - fee,
            -MARGIN
        )

        if status == "LIQUIDATED":

            net = -MARGIN

        trades.append({

            "trade_id": trade_id,

            "tf": tf,

            "side":
                "LONG"
                if pos == 1
                else "SHORT",

            "entry_time": et,

            "entry_price": epx,

            "exit_time": tm,

            "exit_price": px,

            "portion": "100%",

            "gross_pnl": round(
                max(
                    gross,
                    -MARGIN
                ),
                2
            ),

            "fees": round(
                fee,
                2
            ),

            "net_pnl": round(
                net,
                2
            ),

            "status": status

        })


    # =====================================================
    # MAIN LOOP
    # =====================================================

    for i in range(
        2 * length,
        len(df)
    ):

        # =================================================
        # MANAGE CURRENT POSITION
        # =================================================

        if pos:

            # =============================================
            # 1. CHECK 75% TP
            # =============================================

            if not tp_hit:

                if pos == 1:

                    tp_price = (
                        epx
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
                        epx
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

                    close_tp(
                        tp_price,
                        t.iloc[i],
                        i
                    )


            # =============================================
            # 2. BREAKEVEN
            #
            # IMPORTANT:
            # Do NOT check BE on TP candle itself.
            # BE starts from NEXT candle.
            # =============================================

            if (
                pos
                and tp_hit
                and BREAKEVEN_AFTER_TP
                and tp_bar_index is not None
                and i > tp_bar_index
            ):

                if pos == 1:

                    if low[i] <= epx:

                        close_runner(
                            epx,
                            t.iloc[i],
                            "BREAKEVEN"
                        )

                        pos = 0
                        epx = None
                        et = None
                        tp_hit = False
                        tp_bar_index = None

                        continue

                else:

                    if high[i] >= epx:

                        close_runner(
                            epx,
                            t.iloc[i],
                            "BREAKEVEN"
                        )

                        pos = 0
                        epx = None
                        et = None
                        tp_hit = False
                        tp_bar_index = None

                        continue


            # =============================================
            # 3. ORIGINAL SL
            #
            # Only before TP.
            # =============================================

            if pos and not tp_hit and sd:

                if pos == 1:

                    adverse = (
                        epx - low[i]
                    ) / epx

                    if adverse >= sd:

                        stop_price = (
                            epx
                            * (1 - sd)
                        )

                        close_full(
                            stop_price,
                            t.iloc[i],
                            "STOP LOSS"
                        )

                        pos = 0
                        epx = None
                        et = None
                        tp_hit = False
                        tp_bar_index = None

                        continue

                else:

                    adverse = (
                        high[i] - epx
                    ) / epx

                    if adverse >= sd:

                        stop_price = (
                            epx
                            * (1 + sd)
                        )

                        close_full(
                            stop_price,
                            t.iloc[i],
                            "STOP LOSS"
                        )

                        pos = 0
                        epx = None
                        et = None
                        tp_hit = False
                        tp_bar_index = None

                        continue


            # =============================================
            # 4. LIQUIDATION
            # =============================================

            if pos:

                if pos == 1:

                    adverse = (
                        epx - low[i]
                    ) / epx

                    if adverse >= liq:

                        liq_price = (
                            epx
                            * (1 - liq)
                        )

                        if tp_hit:

                            close_runner(
                                liq_price,
                                t.iloc[i],
                                "LIQUIDATED"
                            )

                        else:

                            close_full(
                                liq_price,
                                t.iloc[i],
                                "LIQUIDATED"
                            )

                        pos = 0
                        epx = None
                        et = None
                        tp_hit = False
                        tp_bar_index = None

                        continue

                else:

                    adverse = (
                        high[i] - epx
                    ) / epx

                    if adverse >= liq:

                        liq_price = (
                            epx
                            * (1 + liq)
                        )

                        if tp_hit:

                            close_runner(
                                liq_price,
                                t.iloc[i],
                                "LIQUIDATED"
                            )

                        else:

                            close_full(
                                liq_price,
                                t.iloc[i],
                                "LIQUIDATED"
                            )

                        pos = 0
                        epx = None
                        et = None
                        tp_hit = False
                        tp_bar_index = None

                        continue


        # =================================================
        # PIVOT DETECTION
        # =================================================

        c = i - length

        wh = high[
            c - length:i + 1
        ]

        wl = low[
            c - length:i + 1
        ]

        ph = (
            high[c] == wh.max()
            and
            (wh == high[c]).sum() == 1
        )

        pl = (
            low[c] == wl.min()
            and
            (wl == low[c]).sum() == 1
        )

        if not (ph or pl):

            continue

        if t.iloc[i] < cutoff:

            continue

        npv += 1

        # Pivot HIGH -> SHORT
        # Pivot LOW  -> LONG

        direction = (
            -1
            if ph
            else 1
        )

        if direction == pos:

            continue


        # =================================================
        # OPPOSITE PIVOT
        # =================================================

        if pos:

            if tp_hit:

                close_runner(
                    close[i],
                    t.iloc[i],
                    "closed"
                )

            else:

                close_full(
                    close[i],
                    t.iloc[i],
                    "closed"
                )


        # =================================================
        # OPEN NEW POSITION
        # =================================================

        trade_id += 1

        pos = direction

        epx = close[i]

        et = t.iloc[i]

        tp_hit = False
        tp_bar_index = None


    # =====================================================
    # CLOSE OPEN POSITION AT LAST PRICE
    # =====================================================

    if pos:

        if tp_hit:

            close_runner(
                close[-1],
                t.iloc[-1],
                "open (MTM)"
            )

        else:

            close_full(
                close[-1],
                t.iloc[-1],
                "open (MTM)"
            )


    return (
        pd.DataFrame(trades),
        npv
    )


# =========================================================
# SUMMARY
# =========================================================

def summarize(
    tf,
    tr,
    npv,
    err=""
):

    s = dict(

        tf=tf,

        pivots=npv,

        trades=0,

        long=0,

        short=0,

        wins=0,

        win_pct=0.0,

        net_pnl=0.0,

        realised=0.0,

        open_mtm=0.0,

        liq=0,

        sl_hits=0,

        tp_hits=0,

        breakeven=0,

        fees=0.0,

        profit_factor=0.0,

        max_dd=0.0,

        note=err
    )


    if tr.empty:

        return s


    w = tr[
        tr.net_pnl > 0
    ]

    l = tr[
        tr.net_pnl <= 0
    ]


    eq = pd.concat(

        [
            pd.Series([0.0]),
            tr.net_pnl.cumsum()
        ],

        ignore_index=True
    )


    gross_loss = abs(
        l.net_pnl.sum()
    )


    s.update(

        trades=len(tr),

        long=int(
            (tr.side == "LONG").sum()
        ),

        short=int(
            (tr.side == "SHORT").sum()
        ),

        wins=len(w),

        win_pct=round(
            len(w)
            / len(tr)
            * 100,
            1
        ),

        net_pnl=round(
            tr.net_pnl.sum(),
            2
        ),

        realised=round(
            tr[
                tr.status != "open (MTM)"
            ].net_pnl.sum(),
            2
        ),

        open_mtm=round(
            tr[
                tr.status == "open (MTM)"
            ].net_pnl.sum(),
            2
        ),

        liq=int(
            (
                tr.status
                == "LIQUIDATED"
            ).sum()
        ),

        sl_hits=int(
            (
                tr.status
                == "STOP LOSS"
            ).sum()
        ),

        tp_hits=int(
            (
                tr.status
                == "TP 2%"
            ).sum()
        ),

        breakeven=int(
            (
                tr.status
                == "BREAKEVEN"
            ).sum()
        ),

        fees=round(
            tr.fees.sum(),
            2
        ),

        profit_factor=round(
            w.net_pnl.sum()
            / gross_loss,
            2
        )
        if gross_loss
        else float("inf"),

        max_dd=round(
            (
                eq
                - eq.cummax()
            ).min(),
            2
        )
    )

    return s


# =========================================================
# EQUITY PLOT
# =========================================================

def plot_all(
    all_tr,
    path
):

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:

        return


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    for tf, g in all_tr.groupby(
        "tf",
        sort=False
    ):

        g = g.sort_values(
            "exit_time"
        )

        ax.step(
            g.exit_time,
            g.net_pnl.cumsum(),
            where="post",
            marker="o",
            label=tf
        )


    ax.axhline(
        0,
        color="gray",
        lw=0.8
    )

    ax.set_title(
        "Equity curves - all timeframes (Rs, net of fees)"
    )

    ax.set_xlabel(
        "Exit time"
    )

    ax.set_ylabel(
        "Net P&L (Rs)"
    )

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=120
    )

    plt.close(fig)


# =========================================================
# GITHUB SUMMARY
# =========================================================

def gh_summary(
    sm,
    args,
    pair
):

    path = os.environ.get(
        "GITHUB_STEP_SUMMARY"
    )

    if not path:

        return


    with open(
        path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(

            f"## Pivot Backtest | "
            f"{pair} | "
            f"{args.price} | "
            f"last {args.days} days | "
            f"length {args.length} | "
            f"{LEV:g}x | "
            f"75% TP {TP_PCT:g}% | "
            f"25% runner BE after TP\n\n"
        )


        f.write(

            "| TF | Trades | Win% | Net P&L | "
            "Realised | Open (MTM) | TP hits | "
            "BE | Liquidated | SL hits | Fees | "
            "PF | Max DD |\n"
        )


        f.write(

            "|---|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|\n"
        )


        for r in sm.itertuples():

            f.write(

                f"| {r.tf} | "
                f"{r.trades} | "
                f"{r.win_pct}% | "
                f"{r.net_pnl} | "
                f"{r.realised} | "
                f"{r.open_mtm} | "
                f"{r.tp_hits} | "
                f"{r.breakeven} | "
                f"{r.liq} | "
                f"{r.sl_hits} | "
                f"{r.fees} | "
                f"{r.profit_factor} | "
                f"{r.max_dd} |"
                f"{' ' + r.note if r.note else ''}\n"
            )


        f.write(

            f"\nMargin Rs {MARGIN:.0f} x "
            f"{LEV:g}x = "
            f"Rs {MARGIN * LEV:.0f} position. "
            f"75% ({TP_PART:.0%}) exits at "
            f"+{TP_PCT:g}% price move; "
            f"remaining 25% runner moves SL "
            f"to breakeven from the NEXT candle "
            f"after TP. "
            f"Fee {FEE_PCT}%/side on each portion. "
            f"Funding & slippage ignored.\n"
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser()


    ap.add_argument(
        "--pair",
        default="BTCUSDT"
    )


    ap.add_argument(
        "--days",
        type=int,
        default=30
    )


    ap.add_argument(
        "--price",
        default="LAST_PRICE",
        choices=list(EXCHANGES)
    )


    ap.add_argument(
        "--length",
        type=int,
        default=50
    )


    ap.add_argument(
        "--tfs",
        default=",".join(TFS)
    )


    ap.add_argument(
        "--first",
        default="",
        help="preferred exchange tried first (ccxt id)"
    )


    ap.add_argument(
        "--lev",
        type=float,
        default=10
    )


    ap.add_argument(
        "--margin",
        type=float,
        default=1000
    )


    ap.add_argument(
        "--sl",
        type=float,
        default=0,
        help="stop-loss %% of price, 0 = off"
    )


    a = ap.parse_args()


    LEV = a.lev

    MARGIN = a.margin

    SL = a.sl


    pair = norm_pair(
        a.pair
    )


    all_tr = []

    rows = []


    for tf in a.tfs.split(","):

        tf = tf.strip()

        if not tf:

            continue


        try:

            df = fetch(

                pair,

                tf,

                a.days,

                2 * a.length + 5,

                a.price,

                a.first.strip().lower()
            )


            tr, npv = backtest(

                df,

                a.length,

                a.days,

                tf
            )


            rows.append(

                summarize(

                    tf,

                    tr,

                    npv
                )
            )


            if not tr.empty:

                all_tr.append(
                    tr
                )


        except Exception as e:

            print(
                f"[{tf}] ERROR: {e}"
            )


            rows.append(

                summarize(

                    tf,

                    pd.DataFrame(),

                    0,

                    "ERROR: "
                    + str(e)[:300]
                )
            )


    sm = pd.DataFrame(
        rows
    )


    print(
        "\n"
        + "=" * 90
    )


    print(

        f"{pair} | "
        f"{a.price} | "
        f"last {a.days}d | "
        f"length {a.length} | "
        f"margin Rs {MARGIN:.0f} x "
        f"{LEV:g}x | "
        f"75% TP {TP_PCT:g}% | "
        f"25% runner BE | "
        f"SL {SL:g}% | "
        f"fee {FEE_PCT}%/side"
    )


    print(
        "=" * 90
    )


    print(

        sm.drop(
            columns="note"
        ).to_string(
            index=False
        )
    )


    sm.to_csv(
        "backtest_summary.csv",
        index=False
    )


    if all_tr:

        at = pd.concat(

            all_tr,

            ignore_index=True
        )


        at.to_csv(

            "backtest_all_trades.csv",

            index=False
        )


        print(

            "\nALL TRADES\n"
            + at.to_string(
                index=False
            )
        )


        plot_all(

            at,

            "equity_all.png"
        )


    else:

        pd.DataFrame().to_csv(

            "backtest_all_trades.csv",

            index=False
        )


    gh_summary(

        sm,

        a,

        pair
    )