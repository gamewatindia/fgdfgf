#!/usr/bin/env python3
"""
(VWAP rule: default "reversal" = long only if price was ALREADY below VWAP before
the signal candle, short only if ALREADY above. Use --vwap-mode trend for the opposite.)
BRAHMASTRA STRATEGY - Backtest engine
=====================================
Rules (as described by the user):
 1. Supertrend (ATR 20, mult 2) flips direction and the candle CLOSES beyond the
    line -> this is the "signal/confirmation candle" (C). On a following candle,
    the trade triggers when price breaks the HIGH of C (long) / LOW of C (short).
 2. MACD (12,26,9) crossover in the trade direction. A cross within the last
    `macd_lookback` candles before/at C, or after C (before entry), counts.
 3. Long only if price (last closed candle) is ABOVE VWAP, short only if BELOW.
 Exit:
    - SL  : low of the candle BEFORE the signal candle (long) / its high (short)
    - TP  : Supertrend flips against the trade OR opposite MACD crossover,
            confirmed on candle close (exit at that close).

No look-ahead: every condition is evaluated on candles closed before the entry
candle; the entry itself is a stop-entry filled at max(open, trigger) (long).
If SL and entry occur in the same candle, SL is assumed hit (conservative).

Data: CSV per symbol in a folder, columns: timestamp,open,high,low,close,volume
(timestamp = unix seconds / ms, or ISO string, UTC). Any base timeframe that is
<= the timeframes you test (e.g. 5m data gives 5m/15m/1h/4h/1d/1w/1M).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TF_RULES = {
    "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h",
    "1d": "1D", "1w": "W-MON", "1M": "MS",
}
TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080, "1M": 43200}


# ----------------------------------------------------------------- data ----
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in df.columns if c in ("timestamp", "time", "date", "datetime", "open_time"))
    t = df[tcol]
    if np.issubdtype(t.dtype, np.number):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    df = df.set_index(idx.rename("ts"))[["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df = df.astype(float).sort_index()
    return df[~df.index.duplicated()]


def resample(df, tf):
    rule = TF_RULES[tf]
    kw = dict(label="left", closed="left")
    out = df.resample(rule, **kw).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    return out


# ----------------------------------------------------------- indicators ----
def supertrend(df, period=20, mult=2.0):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values  # Wilder RMA
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    n = len(c)
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == -1 and c[i] > fub[i]:
            d[i] = 1
        elif d[i - 1] == 1 and c[i] < flb[i]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    line = np.where(d == 1, flb, fub)
    return d, line


def macd_cross(close, fast=12, slow=26, sig=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    # "fast line crosses slow line" = MACD line crossing the signal line
    diff = (macd - signal).values
    cross = np.zeros(len(diff), dtype=np.int8)
    cross[1:] = np.where((diff[1:] > 0) & (diff[:-1] <= 0), 1,
                         np.where((diff[1:] < 0) & (diff[:-1] >= 0), -1, 0))
    return cross


def vwap(df, tf):
    """Anchored VWAP. Intraday TF -> resets each UTC day. 1d -> weekly anchor,
    1w -> monthly anchor, 1M -> yearly anchor (daily anchor is meaningless there)."""
    idx = df.index
    if tf in ("5m", "15m", "1h", "4h"):
        key = idx.normalize()
    elif tf == "1d":
        key = idx.tz_localize(None).to_period("W-SUN").start_time
    elif tf == "1w":
        key = idx.tz_localize(None).to_period("M").start_time
    else:
        key = idx.tz_localize(None).to_period("Y").start_time
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).groupby(key).cumsum()
    vv = df["volume"].groupby(key).cumsum()
    return (pv / vv.replace(0, np.nan)).fillna(tp).values


# ------------------------------------------------------------- backtest ----
def backtest(df, tf, p):
    n = len(df)
    if n < 60:
        return []
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    ts = df.index
    d, st_line = supertrend(df, p["st_period"], p["st_mult"])
    mc = macd_cross(df["close"])
    vw = vwap(df, tf)
    slip, fee = p["slip"] / 100, p["fee"] / 100
    lb, wait = p["macd_lookback"], p["wait_bars"]

    trades, pos = [], None
    setups = {1: None, -1: None}
    start = max(p["st_period"], 30) + 2

    def close_trade(i, px, why):
        nonlocal pos
        s = pos["side"]
        xpx = px * (1 - slip * s)
        gross = s * (xpx / pos["entry"] - 1)
        cost = 2 * fee + (p["tds"] / 100 if True else 0)  # TDS on sell leg (set 0 if n/a)
        net = gross - cost
        risk = abs(pos["entry"] - pos["sl"]) / pos["entry"]
        trades.append(dict(
            side="LONG" if s == 1 else "SHORT", entry_time=ts[pos["i"]], exit_time=ts[i],
            entry=pos["entry"], exit=xpx, sl=pos["sl"], exit_reason=why,
            bars=i - pos["i"], gross_pct=gross * 100, net_pct=net * 100,
            R=(net / risk) if risk > 0 else np.nan))
        pos = None

    for i in range(start, n):
        # ---- invalidate / expire setups using info up to bar i-1
        for s in (1, -1):
            st = setups[s]
            if st and (d[i - 1] != s or i > st["C"] + wait):
                setups[s] = None

        # ---- try entry (stop-entry on bar i), only one position at a time
        if pos is None:
            for s in (1, -1):
                st = setups[s]
                if not st or i <= st["C"]:
                    continue
                if s == 1 and not p["allow_long"] or s == -1 and not p["allow_short"]:
                    continue
                if not (st["trigger"] < h[i] if s == 1 else st["trigger"] > l[i]):
                    continue
                # condition 2: MACD cross in direction within window [C-lb, i-1]
                if not (mc[max(0, st["C"] - lb): i] == s).any():
                    continue
                # condition 3: VWAP side. "already" mode: price must ALREADY be on the
                # right side of VWAP before the signal candle (candle C-1) and still
                # be there on the last closed candle before entry.
                if p["vwap_mode"] == "reversal":
                    # price was ALREADY below VWAP (long) / above VWAP (short) before the signal candle
                    k = st["C"] - 1
                    ok = c[k] < vw[k] if s == 1 else c[k] > vw[k]
                else:  # "trend": long above VWAP / short below VWAP, checked before entry
                    ok = c[i - 1] > vw[i - 1] if s == 1 else c[i - 1] < vw[i - 1]
                if not ok:
                    continue
                raw = max(o[i], st["trigger"]) if s == 1 else min(o[i], st["trigger"])
                entry = raw * (1 + slip * s)
                sl = st["sl"]
                if (sl >= entry if s == 1 else sl <= entry):
                    sl = st["sl_alt"]
                if (sl >= entry if s == 1 else sl <= entry):
                    continue
                pos = dict(side=s, i=i, entry=entry, sl=sl)
                setups[s] = None
                break

        # ---- manage open position on bar i
        if pos is not None:
            s = pos["side"]
            hit = l[i] <= pos["sl"] if s == 1 else h[i] >= pos["sl"]
            if hit:
                fill = min(o[i], pos["sl"]) if s == 1 and i > pos["i"] else \
                       max(o[i], pos["sl"]) if s == -1 and i > pos["i"] else pos["sl"]
                close_trade(i, fill, "SL")
            elif d[i] == -s and d[i - 1] == s:
                close_trade(i, c[i], "TP_SUPERTREND")
            elif mc[i] == -s:
                close_trade(i, c[i], "TP_MACD")

        # ---- arm new setup at close of bar i (supertrend flip + close beyond line)
        if d[i] != d[i - 1]:
            s = int(d[i])
            if i >= 1:
                setups[s] = dict(
                    C=i,
                    trigger=h[i] if s == 1 else l[i],
                    sl=l[i - 1] if s == 1 else h[i - 1],   # candle before signal candle
                    sl_alt=l[i] if s == 1 else h[i])
                setups[-s] = None
    if pos is not None:                      # mark-to-market final bar
        close_trade(n - 1, c[-1], "END")
    return trades


def stats(trades, df, alloc):
    if not trades:
        return dict(trades=0)
    t = pd.DataFrame(trades)
    r = t["net_pct"].values / 100 * alloc
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    dd = (eq / peak - 1).min() * 100
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    wins, losses = t.loc[t.net_pct > 0, "net_pct"], t.loc[t.net_pct <= 0, "net_pct"]
    return dict(
        trades=len(t), win_rate=round((t.net_pct > 0).mean() * 100, 1),
        total_return_pct=round((eq[-1] - 1) * 100, 2),
        cagr_pct=round(((eq[-1]) ** (1 / years) - 1) * 100, 2) if eq[-1] > 0 else -100,
        max_dd_pct=round(dd, 2),
        profit_factor=round(wins.sum() / abs(losses.sum()), 2) if len(losses) and losses.sum() != 0 else np.inf,
        avg_trade_pct=round(t.net_pct.mean(), 3), avg_R=round(t.R.mean(), 2),
        buy_hold_pct=round((df.close.iloc[-1] / df.close.iloc[0] - 1) * 100, 2),
    ), eq


def run(data_dir, tfs, p, outdir):
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        sys.exit(f"No CSV files in {data_dir}")
    rows, all_tr, curves = [], [], {}
    for f in files:
        sym = os.path.splitext(os.path.basename(f))[0]
        base = load_csv(f)
        base_min = int(np.median(np.diff(base.index.values).astype("timedelta64[m]").astype(int)))
        for tf in tfs:
            if TF_MINUTES[tf] < base_min:
                continue
            df = resample(base, tf)
            tr = backtest(df, tf, p)
            res = stats(tr, df, p["alloc"])
            if isinstance(res, tuple):
                s, eq = res
                curves[(sym, tf)] = (pd.Series(eq, index=[t["exit_time"] for t in tr]))
            else:
                s = res
            rows.append(dict(symbol=sym, tf=tf, bars=len(df), **s))
            for t in tr:
                all_tr.append(dict(symbol=sym, tf=tf, **t))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    pd.DataFrame(all_tr).to_csv(os.path.join(outdir, "trades.csv"), index=False)
    return summ, curves


def main():
    ap = argparse.ArgumentParser(description="Brahmastra strategy backtest")
    ap.add_argument("--data", default="data", help="folder with SYMBOL.csv files")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tfs", default="5m,15m,1h,4h,1d,1w,1M")
    ap.add_argument("--st-period", type=int, default=20)
    ap.add_argument("--st-mult", type=float, default=2.0)
    ap.add_argument("--macd-lookback", type=int, default=2, help="candles before signal candle to accept a MACD cross")
    ap.add_argument("--wait-bars", type=int, default=3, help="candles after signal candle in which breakout may trigger (1 = strictly next candle)")
    ap.add_argument("--fee", type=float, default=0.1, help="%% per side")
    ap.add_argument("--slip", type=float, default=0.05, help="%% per side")
    ap.add_argument("--tds", type=float, default=0.0, help="%% India crypto TDS on sell (1.0 for real-world)")
    ap.add_argument("--alloc", type=float, default=1.0, help="fraction of equity per trade (1.0 = all-in, no leverage)")
    ap.add_argument("--vwap-mode", choices=["reversal", "trend"], default="reversal",
                    help="reversal (default): long if price was already BELOW VWAP before the signal candle, short if already ABOVE; "
                         "trend: long if price above VWAP, short if below")
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--no-long", action="store_true")
    a = ap.parse_args()
    p = dict(st_period=a.st_period, st_mult=a.st_mult, macd_lookback=a.macd_lookback,
             wait_bars=a.wait_bars, vwap_mode=a.vwap_mode, fee=a.fee, slip=a.slip, tds=a.tds, alloc=a.alloc,
             allow_long=not a.no_long, allow_short=not a.no_short)
    summ, curves = run(a.data, [t.strip() for t in a.tfs.split(",")], p, a.out)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summ.to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = summ.dropna(subset=["total_return_pct"]).sort_values("total_return_pct", ascending=False).head(6)
        fig, ax = plt.subplots(figsize=(10, 5))
        for _, r in top.iterrows():
            s = curves.get((r.symbol, r.tf))
            if s is not None:
                ax.plot(s.index, (s.values - 1) * 100, label=f"{r.symbol} {r.tf}")
        ax.set_ylabel("Return % (compounded, per-trade)")
        ax.set_title("Brahmastra - top equity curves")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "equity_top.png"), dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
