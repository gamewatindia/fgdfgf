"""
Pivot High/Low (LuxAlgo-style) Backtest - ALL TIMEFRAMES in ONE run

MODES (--mode)
  reverse              Stop-and-reverse. Entry = close of the bar where the pivot is CONFIRMED
                       (50 bars after the real pivot). Tradable, no look-ahead.
  reverse_pivot_price  Same, but entry = the real pivot price on the pivot bar.
                       NOT TRADABLE (needs future data) - only to compare how much the lag costs.
  hold_long / hold_short / hold_both
                       Every confirmed pivot opens a NEW position (long = pivot lows, short = pivot
                       highs). No stop-loss, opposite signals are ignored, each position is held until
                       its take-profit (--tp %). CROSS margin: all positions share one account
                       (--balance). If account equity can't cover the open losses, the WHOLE account
                       is liquidated.
                       --max_open N : at most N positions at the same time (extra signals skipped).
                       --min_open N : keep at least N positions running (a take-profit that would drop
                                      the count below N is postponed until a new position opens).

Rs MARGIN per trade x LEV leverage. Fee on full position size. Liquidation (isolated, reverse modes):
~ 1/LEV - 0.5% adverse move = full margin lost. Optional --sl / --tp in % of PRICE
(10x: 5% price = 50% on margin).

Every output file name + every row carries run_id / coin / pair / exchange so uploads are self-explaining.
Outputs (folder results/): <run_id>_summary.csv, _trades.csv, _pivots.csv, _equity.png, <run_id>.xlsx
Run: python pivot_backtest.py --pair BTCUSDT --days 30 --mode hold_long --tp 5 --balance 10000 --max_open 3
Pair can carry a preferred exchange:  --pair BRUSDT@mexc
"""
import argparse, os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

MARGIN = 1000.0             # Rs margin per trade
LEV = 10                    # leverage
MMR = 0.005                 # maintenance margin (0.5%)
FEE_PCT = 0.10              # % per side, on full position size
SL = 0.0                    # stop-loss in % of price (0 = off)
TP = 0.0                    # take-profit in % of price (0 = off)
TFS = ["5m", "15m", "30m", "1h", "4h"]
MODES = ["reverse", "reverse_pivot_price", "hold_long", "hold_short", "hold_both"]

# Tried in order until one has the pair. (bybit/binance.com block GitHub's US servers,
# so they are kept last; gateio/mexc/kucoin/bitget list far more altcoins.)
EXCHANGES = {
    "LAST_PRICE": ["binanceus", "gateio", "mexc", "kucoin", "bitget", "okx", "bybit"],
    "MARK_PRICE": ["gateio", "bitget", "okx", "bybit", "binanceusdm", "kucoinfutures"],
    "INDEX_PRICE": ["gateio", "bitget", "okx", "bybit", "binanceusdm"],
}

HOLD_ONLY_COLS = [
    "account_liq",
    "min_equity",
    "final_equity",
    "skipped",
    "skipped_cap",
    "peak_open"
]


def norm_pair(p):
    p = p.upper().replace("-", "/").replace("_", "/")
    if "/" not in p:
        for q in ("USDT", "USDC", "USD"):
            if p.endswith(q):
                return p[: -len(q)] + "/" + q
    return p


def fetch(pair, tf, days, warmup, price, first=""):
    """Returns (df, info). Tries exchanges in order. Prefers one with the FULL requested history;
    if none has it, uses the one with the most history and prints a warning."""
    import ccxt

    errs, best = [], None
    order = ([first] if first else []) + [
        e for e in EXCHANGES[price] if e != first
    ]

    for name in order:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()

            sym = pair if price == "LAST_PRICE" else pair + ":" + pair.split("/")[1]

            if sym not in ex.markets:
                raise ValueError(f"{sym} not listed")

            fn = getattr(
                ex,
                {
                    "LAST_PRICE": "fetch_ohlcv",
                    "MARK_PRICE": "fetch_mark_ohlcv",
                    "INDEX_PRICE": "fetch_index_ohlcv"
                }[price]
            )

            tf_ms = ex.parse_timeframe(tf) * 1000
            now = ex.milliseconds()
            since = now - days * 86400000 - warmup * tf_ms

            rows = []

            while since < now:
                b = fn(sym, tf, since=since, limit=1000)

                if not b:
                    break

                rows += b

                nxt = b[-1][0] + tf_ms

                if nxt <= since:
                    break

                since = nxt
                time.sleep(ex.rateLimit / 1000)

            if len(rows) < 2 * warmup:
                raise ValueError(f"only {len(rows)} candles")

            df = pd.DataFrame(rows).iloc[:, :5]
            df.columns = ["time", "open", "high", "low", "close"]

            df["time"] = pd.to_datetime(df["time"], unit="ms")

            df = (
                df.drop_duplicates("time")
                .sort_values("time")
                .reset_index(drop=True)
            )

            span = (
                df["time"].iloc[-1] - df["time"].iloc[0]
            ).total_seconds() / 86400

            covered = span - warmup * tf_ms / 86400000

            info = dict(
                exchange=name,
                symbol=sym,
                candles=len(df),
                days_covered=round(covered),
                data_from=str(df["time"].iloc[0]),
                data_to=str(df["time"].iloc[-1])
            )

            if covered >= days * 0.97:
                print(
                    f"[{tf}] data: {name} {sym} {price} "
                    f"({len(df)} candles, {covered:.0f} days)"
                )
                return df, info

            errs.append(f"{name}: only {covered:.0f} of {days} days")

            if best is None or covered > best[0]:
                best = (covered, name, sym, df, info)

        except Exception as e:
            errs.append(f"{name}: {str(e)[:60]}")

    if best:
        covered, name, sym, df, info = best

        print(
            f"[{tf}] WARNING: no exchange had {days} days. "
            f"Using {name} {sym}: only {covered:.0f} days "
            f"({len(df)} candles). Others: {' | '.join(errs)}"
        )

        return df, info

    raise RuntimeError(" | ".join(errs))


def get_signals(df, length, days, at_pivot):
    """{bar_index: (direction, price)}. direction +1 = pivot low (long), -1 = pivot high (short).
    at_pivot=False: signal on the CONFIRMATION bar, price = its close (tradable).
    at_pivot=True : signal on the real pivot bar, price = pivot extreme (look-ahead)."""

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    t = df["time"]

    cutoff = t.iloc[-1] - pd.Timedelta(days=days)
    sig = {}

    for i in range(2 * length, len(df)):
        c = i - length

        wh = high[c - length: i + 1]
        wl = low[c - length: i + 1]

        ph = high[c] == wh.max() and (wh == high[c]).sum() == 1
        pl = low[c] == wl.min() and (wl == low[c]).sum() == 1

        if not (ph or pl):
            continue

        k = c if at_pivot else i

        if t.iloc[k] < cutoff:
            continue

        px = (
            (high[c] if ph else low[c])
            if at_pivot
            else close[i]
        )

        sig[k] = (-1 if ph else 1, px)

    return sig


def excursion(d, epx, hi, lo):
    """(adverse, favourable) move of this bar vs entry, fraction of price. adverse > 0 = against you."""
    if d == 1:
        return (epx - lo) / epx, (hi - epx) / epx

    return (hi - epx) / epx, (epx - lo) / epx


def list_pivots(df, length, days, tf):
    """Every confirmed pivot: when it really happened vs when the strategy could act on it."""

    high = df["high"].values
    low = df["low"].values
    t = df["time"]

    cutoff = t.iloc[-1] - pd.Timedelta(days=days)

    out = []

    for i in range(2 * length, len(df)):
        c = i - length

        wh = high[c - length: i + 1]
        wl = low[c - length: i + 1]

        ph = high[c] == wh.max() and (wh == high[c]).sum() == 1
        pl = low[c] == wl.min() and (wl == low[c]).sum() == 1

        if (ph or pl) and t.iloc[c] >= cutoff:
            out.append(
                dict(
                    tf=tf,
                    type="PIVOT HIGH (short)" if ph else "PIVOT LOW (long)",
                    pivot_time=t.iloc[c],
                    pivot_price=high[c] if ph else low[c],
                    confirmed_time=t.iloc[i],
                    entry_price_if_traded=df["close"].iloc[i]
                )
            )

    return pd.DataFrame(out)


def reverse_backtest(df, length, days, tf, at_pivot):
    high = df["high"].values
    low = df["low"].values
    t = df["time"]

    sig = get_signals(df, length, days, at_pivot)

    notional = MARGIN * LEV

    liq = max(
        1 / LEV - MMR,
        0.001
    )

    sd = SL / 100 if 0 < SL / 100 < liq else None
    tp = TP / 100 if TP > 0 else None

    trades = []
    pos = 0
    epx = None
    et = None
    mae = 0.0
    mfe = 0.0

    def close_trade(px, tm, status):
        g = notional * pos * (px / epx - 1)

        fee = notional * FEE_PCT / 100 * 2

        net = max(
            g - fee,
            -MARGIN
        )

        if status == "LIQUIDATED":
            net = -MARGIN

        trades.append(
            dict(
                tf=tf,
                side="LONG" if pos == 1 else "SHORT",
                entry_time=et,
                entry_price=epx,
                exit_time=tm,
                exit_price=px,
                mfe_pct=round(mfe * 100, 2),
                mae_pct=round(mae * 100, 2),
                hold_h=round(
                    (tm - et).total_seconds() / 3600,
                    1
                ),
                gross_pnl=round(
                    max(g, -MARGIN),
                    2
                ),
                fees=round(fee, 2),
                net_pnl=round(net, 2),
                status=status
            )
        )

    for j in range(len(df)):
        if pos:
            adv, fav = excursion(
                pos,
                epx,
                high[j],
                low[j]
            )

            mae = max(mae, adv)
            mfe = max(mfe, fav)

            if sd and adv >= sd:
                close_trade(
                    epx * (1 - pos * sd),
                    t.iloc[j],
                    "STOP LOSS"
                )
                pos = 0

            elif adv >= liq:
                close_trade(
                    epx * (1 - pos * liq),
                    t.iloc[j],
                    "LIQUIDATED"
                )
                pos = 0

            elif tp and fav >= tp:
                close_trade(
                    epx * (1 + pos * tp),
                    t.iloc[j],
                    "TARGET"
                )
                pos = 0

        if j in sig:
            d, px = sig[j]

            if d == pos:
                continue

            if pos:
                close_trade(
                    px,
                    t.iloc[j],
                    "closed"
                )

            pos = d
            epx = px
            et = t.iloc[j]
            mae = 0.0
            mfe = 0.0

    if pos:
        close_trade(
            df["close"].iloc[-1],
            t.iloc[-1],
            "open (MTM)"
        )

    return pd.DataFrame(trades), len(sig), {}


def hold_backtest(
    df,
    length,
    days,
    tf,
    side,
    balance,
    min_open=0,
    max_open=0
):
    """Cross margin, no SL, opposite signals ignored, every position held until TP.
    max_open: cap on simultaneous positions (0 = no cap).
    min_open: keep at least this many running; a TP that would go below it is postponed."""

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    t = df["time"]

    sig = get_signals(
        df,
        length,
        days,
        False
    )

    notional = MARGIN * LEV
    fee_side = notional * FEE_PCT / 100
    tp = TP / 100

    cash = balance
    opn = []
    trades = []
    skipped = 0
    dead = False
    min_eq = balance

    skipped_cap = 0
    peak = 0

    def rec(p, px, tm, status):
        g = (
            notional
            * p["d"]
            * (px / p["epx"] - 1)
        )

        trades.append(
            dict(
                tf=tf,
                side="LONG" if p["d"] == 1 else "SHORT",
                entry_time=p["et"],
                entry_price=p["epx"],
                exit_time=tm,
                exit_price=px,
                mfe_pct=round(p["mfe"] * 100, 2),
                mae_pct=round(
                    max(p["mae"], 0) * 100,
                    2
                ),
                hold_h=round(
                    (tm - p["et"]).total_seconds() / 3600,
                    1
                ),
                gross_pnl=round(g, 2),
                fees=round(2 * fee_side, 2),
                net_pnl=round(
                    g - 2 * fee_side,
                    2
                ),
                status=status
            )
        )

        return g

    for j in range(len(df)):
        if opn:
            worst = 0.0

            for p in opn:
                adv, fav = excursion(
                    p["d"],
                    p["epx"],
                    high[j],
                    low[j]
                )

                p["mae"] = max(
                    p["mae"],
                    adv
                )

                p["mfe"] = max(
                    p["mfe"],
                    fav
                )

                p["fav"] = fav

                worst -= (
                    adv
                    * notional
                )

            eq = cash + worst

            min_eq = min(
                min_eq,
                eq
            )

            if eq <= len(opn) * notional * MMR:
                for p in opn:
                    rec(
                        p,
                        low[j] if p["d"] == 1 else high[j],
                        t.iloc[j],
                        "ACCOUNT LIQUIDATED"
                    )

                opn = []
                cash = 0.0
                dead = True

                break

            for p in list(opn):
                hit = p["fav"] >= tp

                if hit or p.get("due"):
                    if len(opn) <= min_open:
                        p["due"] = True
                        continue

                    if hit:
                        g = rec(
                            p,
                            p["epx"] * (1 + p["d"] * tp),
                            t.iloc[j],
                            "TARGET"
                        )
                    else:
                        g = rec(
                            p,
                            close[j],
                            t.iloc[j],
                            "DEFERRED EXIT (min_open)"
                        )

                    cash += g - fee_side
                    opn.remove(p)

        if j in sig:
            d, px = sig[j]

            if side and d != side:
                continue

            if max_open and len(opn) >= max_open:
                skipped_cap += 1
                continue

            unreal = sum(
                notional
                * p["d"]
                * (close[j] / p["epx"] - 1)
                for p in opn
            )

            if (
                cash
                + unreal
                - len(opn) * MARGIN
                < MARGIN + fee_side
            ):
                skipped += 1
                continue

            cash -= fee_side

            opn.append(
                dict(
                    d=d,
                    epx=px,
                    et=t.iloc[j],
                    mae=0.0,
                    mfe=0.0,
                    fav=0.0
                )
            )

            peak = max(
                peak,
                len(opn)
            )

    unreal = 0.0

    for p in opn:
        rec(
            p,
            close[-1],
            t.iloc[-1],
            "open (MTM)"
        )

        unreal += (
            notional
            * p["d"]
            * (close[-1] / p["epx"] - 1)
        )

    extra = dict(
        account_liq=dead,
        min_equity=round(
            min_eq if not dead else 0.0,
            2
        ),
        final_equity=round(
            cash + unreal,
            2
        ),
        skipped=skipped,
        skipped_cap=skipped_cap,
        peak_open=peak
    )

    return (
        pd.DataFrame(trades),
        len(sig),
        extra
    )


def summarize(
    tf,
    tr,
    npv,
    extra=None,
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
        tp_hits=0,
        open_n=0,
        liq=0,
        sl_hits=0,
        worst_mae=0.0,
        fees=0.0,
        profit_factor=0.0,
        max_dd=0.0,
        account_liq="",
        min_equity="",
        final_equity="",
        skipped="",
        skipped_cap="",
        peak_open="",
        note=err
    )

    s.update(extra or {})

    if tr.empty:
        return s

    w = tr[tr.net_pnl > 0]
    l = tr[tr.net_pnl <= 0]

    eq = pd.concat(
        [
            pd.Series([0.0]),
            tr.net_pnl.cumsum()
        ],
        ignore_index=True
    )

    gl = abs(
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
            len(w) / len(tr) * 100,
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
        tp_hits=int(
            (tr.status == "TARGET").sum()
        ),
        open_n=int(
            (tr.status == "open (MTM)").sum()
        ),
        liq=int(
            tr.status
            .str.contains("LIQUIDATED")
            .sum()
        ),
        sl_hits=int(
            (tr.status == "STOP LOSS").sum()
        ),
        worst_mae=round(
            tr.mae_pct.max(),
            2
        ),
        fees=round(
            tr.fees.sum(),
            2
        ),
        profit_factor=round(
            w.net_pnl.sum() / gl,
            2
        ) if gl else float("inf"),
        max_dd=round(
            (eq - eq.cummax()).min(),
            2
        )
    )

    return s


def plot_all(all_tr, path, title):
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
        title + "\nEquity curves (Rs, net of fees)",
        fontsize=10
    )

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=120
    )


LEGEND = [
    (
        "run_id",
        "Unique name of this run: coin-mode-days-leverage-(tp/sl/limits)-UTC time"
    ),
    (
        "coin / pair / exchange",
        "Coin, trading pair, and the exchange the candl
