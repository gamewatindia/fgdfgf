#!/usr/bin/env python3
"""
BRAHMASTRA STRATEGY - Backtest engine (futures style: fixed margin + leverage)
==============================================================================
Rules:
 1. Supertrend (ATR 20, mult 2) flips direction and the candle CLOSES beyond the
    line -> this is the "signal candle" (C). On a following candle the trade
    triggers when price breaks the HIGH of C (long) / LOW of C (short).
 2. MACD (12,26,9) crossover in the trade direction, within `macd_lookback`
    candles before C, at C, or after C (before entry).
 3. VWAP (default "reversal"): long only if price was ALREADY BELOW VWAP before
    the signal candle, short only if ALREADY ABOVE ("trend" mode = opposite).
 Exit:
    - SL  : low of the candle BEFORE the signal candle (long) / its high (short)
    - TP  : Supertrend flips against the trade OR opposite MACD crossover,
            confirmed on candle close (exit at that close).
    - LIQUIDATION: if price reaches the liquidation price before the SL, the
            whole margin is lost.

Money model: every trade uses a fixed MARGIN (default Rs 1000, no compounding).
Position size = margin x leverage. P&L(Rs) = margin x leverage x net price move,
where net price move already includes fees + slippage on the full notional.
Liquidation price = entry x (1 -/+ (1/leverage - maintenance_margin)).

No look-ahead: conditions use candles closed before the entry candle; entry is a
stop-entry filled at max(open, trigger) (long). If SL and entry occur in the same
candle, SL is assumed hit (conservative).

Data: CSV per symbol in a folder, columns: timestamp,open,high,low,close,volume
(timestamp = unix seconds / ms, or ISO string, UTC).
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
    return df.resample(TF_RULES[tf], label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])


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
    return d, np.where(d == 1, flb, fub)


def macd_cross(close, fast=12, slow=26, sig=9):
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal = macd.ewm(span=sig, adjust=False).mean()
    diff = (macd - signal).values  # "fast line crosses slow line" = MACD line vs signal line
    cross = np.zeros(len(diff), dtype=np.int8)
    cross[1:] = np.where((diff[1:] > 0) & (diff[:-1] <= 0), 1,
                         np.where((diff[1:] < 0) & (diff[:-1] >= 0), -1, 0))
    return cross


def vwap(df, tf):
    """Anchored VWAP. Intraday TF -> resets each UTC day. 1d -> weekly anchor,
    1w -> monthly anchor, 1M -> yearly anchor."""
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
    d, _ = supertrend(df, p["st_period"], p["st_mult"])
    mc = macd_cross(df["close"])
    vw = vwap(df, tf)
    slip, fee, tds = p["slip"] / 100, p["fee"] / 100, p["tds"] / 100
    margin, mmr = p["margin"], p["mmr"] / 100
    lb, wait = p["macd_lookback"], p["wait_bars"]

    trades, pos = [], None
    setups = {1: None, -1: None}
    start = max(p["st_period"], 30) + 2

    def close_trade(i, px, why, liq=False):
        nonlocal pos
        s, lev = pos["side"], pos["lev"]
        if liq:
            xpx, net, pnl = px, -1.0 / lev, -margin
        else:
            xpx = px * (1 - slip * s)
            net = s * (xpx / pos["entry"] - 1) - 2 * fee - tds   # costs on full notional
            pnl = max(margin * lev * net, -margin)               # can never lose more than margin
        risk = abs(pos["entry"] - pos["sl"]) / pos["entry"]
        trades.append(dict(
            side="LONG" if s == 1 else "SHORT", lev=lev,
            entry_time=ts[pos["i"]], exit_time=ts[i], entry=pos["entry"], exit=xpx,
            sl=pos["sl"], liq=pos["liq"], exit_reason=why, bars=i - pos["i"],
            sl_pct=risk * 100, price_net_pct=net * 100, pnl_inr=pnl,
            roi_margin_pct=pnl / margin * 100, R=(net / risk) if risk > 0 else np.nan))
        pos = None

    for i in range(start, n):
        for s in (1, -1):                       # expire/invalidate setups (info up to bar i-1)
            st = setups[s]
            if st and (d[i - 1] != s or i > st["C"] + wait):
                setups[s] = None

        if pos is None:                         # ---- try stop-entry on bar i
            for s in (1, -1):
                st = setups[s]
                if not st or i <= st["C"]:
                    continue
                if (s == 1 and not p["allow_long"]) or (s == -1 and not p["allow_short"]):
                    continue
                if not (st["trigger"] < h[i] if s == 1 else st["trigger"] > l[i]):
                    continue
                if not (mc[max(0, st["C"] - lb): i] == s).any():      # MACD cross window
                    continue
                if p["vwap_mode"] == "reversal":                      # VWAP side
                    k = st["C"] - 1
                    ok = c[k] < vw[k] if s == 1 else c[k] > vw[k]
                else:
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
                lev = p["lev_long"] if s == 1 else p["lev_short"]
                liq = entry * (1 - s * max(1.0 / lev - mmr, 1e-9))
                pos = dict(side=s, i=i, entry=entry, sl=sl, lev=lev, liq=liq)
                setups[s] = None
                break

        if pos is not None:                     # ---- manage open position on bar i
            s = pos["side"]
            if s == 1:
                stop = max(pos["sl"], pos["liq"])
                hit, is_liq = l[i] <= stop, pos["liq"] >= pos["sl"]
            else:
                stop = min(pos["sl"], pos["liq"])
                hit, is_liq = h[i] >= stop, pos["liq"] <= pos["sl"]
            if hit:
                if is_liq:
                    close_trade(i, stop, "LIQUIDATION", liq=True)
                else:
                    gap = i > pos["i"]
                    fill = (min(o[i], stop) if s == 1 else max(o[i], stop)) if gap else stop
                    close_trade(i, fill, "SL")
            elif d[i] == -s and d[i - 1] == s:
                close_trade(i, c[i], "TP_SUPERTREND")
            elif mc[i] == -s:
                close_trade(i, c[i], "TP_MACD")

        if d[i] != d[i - 1]:                    # ---- arm new setup at close of bar i
            s = int(d[i])
            setups[s] = dict(C=i, trigger=h[i] if s == 1 else l[i],
                             sl=l[i - 1] if s == 1 else h[i - 1],   # candle before signal candle
                             sl_alt=l[i] if s == 1 else h[i])
            setups[-s] = None
    if pos is not None:
        close_trade(n - 1, c[-1], "END")
    return trades


def stats(trades, df):
    if not trades:
        return dict(trades=0), None
    t = pd.DataFrame(trades)
    pnl = t["pnl_inr"].values
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return dict(
        trades=len(t), win_rate=round((pnl > 0).mean() * 100, 1),
        total_pnl_inr=round(pnl.sum(), 1), avg_pnl_inr=round(pnl.mean(), 1),
        max_dd_inr=round((cum - peak).min(), 1),
        profit_factor=round(gains / losses, 2) if losses > 0 else np.inf,
        avg_roi_margin_pct=round(t.roi_margin_pct.mean(), 2),
        avg_sl_pct=round(t.sl_pct.mean(), 3), avg_R=round(t.R.mean(), 2),
        liquidations=int((t.exit_reason == "LIQUIDATION").sum()),
        buy_hold_pct=round((df.close.iloc[-1] / df.close.iloc[0] - 1) * 100, 2),
    ), pd.Series(cum, index=t["exit_time"].values)


def run(data_dir, tfs, p, scenarios, outdir):
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        sys.exit(f"No CSV files in {data_dir}")
    rows, all_tr, curves = [], [], {}
    for f in files:
        sym = os.path.splitext(os.path.basename(f))[0]
        base = load_csv(f)
        base_min = int(np.median(np.diff(base.index.values).astype("timedelta64[m]").astype(int)))
        span = f"{base.index[0]:%Y-%m-%d} -> {base.index[-1]:%Y-%m-%d}"
        print(f"[data] {sym}: {len(base)} candles ({base_min}m), {span}")
        for tf in tfs:
            if TF_MINUTES[tf] < base_min:
                continue
            df = resample(base, tf)
            for ll, sl_ in scenarios:
                q = dict(p, lev_long=ll, lev_short=sl_)
                tr = backtest(df, tf, q)
                s, cum = stats(tr, df)
                lev_name = f"{ll}x" if ll == sl_ else f"L{ll}x/S{sl_}x"
                rows.append(dict(symbol=sym, tf=tf, lev=lev_name, bars=len(df), **s))
                if cum is not None:
                    curves[(sym, tf, lev_name)] = cum
                for t in tr:
                    all_tr.append(dict(symbol=sym, tf=tf, **t))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    pd.DataFrame(all_tr).to_csv(os.path.join(outdir, "trades.csv"), index=False)
    return summ, curves


def main():
    ap = argparse.ArgumentParser(description="Brahmastra strategy backtest (margin + leverage)")
    ap.add_argument("--config", help="YAML settings file (see config.yml); CLI flags override it")
    ap.add_argument("--data", default="data", help="folder with SYMBOL.csv files")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tfs", default="5m,15m,1h,4h,1d,1w,1M")
    ap.add_argument("--st-period", type=int, default=20)
    ap.add_argument("--st-mult", type=float, default=2.0)
    ap.add_argument("--macd-lookback", type=int, default=2)
    ap.add_argument("--wait-bars", type=int, default=3, help="candles after signal candle in which breakout may trigger")
    ap.add_argument("--vwap-mode", choices=["reversal", "trend"], default="reversal")
    ap.add_argument("--margin", type=float, default=1000.0, help="Rs margin per trade")
    ap.add_argument("--leverages", default="5,10", help="comma list; each value is run as a separate scenario (both sides)")
    ap.add_argument("--long-lev", type=float, help="optional: fixed long leverage (with --short-lev) instead of --leverages")
    ap.add_argument("--short-lev", type=float, help="optional: fixed short leverage")
    ap.add_argument("--mmr", type=float, default=0.5, help="%% maintenance margin used for liquidation price")
    ap.add_argument("--fee", type=float, default=0.02, help="%% per side on notional (futures taker ~0.02-0.05; spot ~0.1)")
    ap.add_argument("--slip", type=float, default=0.02, help="%% per side")
    ap.add_argument("--tds", type=float, default=0.0, help="%% extra cost per trade on notional (0 for futures)")
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--no-long", action="store_true")
    pre, _ = ap.parse_known_args()
    if pre.config:
        import yaml
        with open(pre.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        g = lambda sec, key: (cfg.get(sec) or {}).get(key)
        levs = g("leverage", "leverages")
        mapping = {
            "data": g("data", "folder"), "out": g("backtest", "output_folder"),
            "tfs": ",".join(g("backtest", "timeframes") or []) or None,
            "st_period": g("strategy", "supertrend_period"), "st_mult": g("strategy", "supertrend_multiplier"),
            "macd_lookback": g("strategy", "macd_lookback"), "wait_bars": g("strategy", "wait_bars"),
            "vwap_mode": g("strategy", "vwap_mode"),
            "no_long": (g("strategy", "allow_long") is False) or None,
            "no_short": (g("strategy", "allow_short") is False) or None,
            "fee": g("costs", "fee_pct"), "slip": g("costs", "slippage_pct"), "tds": g("costs", "tds_pct"),
            "margin": g("leverage", "margin_inr"), "mmr": g("leverage", "maintenance_margin_pct"),
            "leverages": ",".join(str(x) for x in levs) if levs else None,
        }
        ap.set_defaults(**{k: v for k, v in mapping.items() if v is not None})
    a = ap.parse_args()

    p = dict(st_period=a.st_period, st_mult=a.st_mult, macd_lookback=a.macd_lookback,
             wait_bars=a.wait_bars, vwap_mode=a.vwap_mode, fee=a.fee, slip=a.slip, tds=a.tds,
             margin=a.margin, mmr=a.mmr, allow_long=not a.no_long, allow_short=not a.no_short)
    if a.long_lev or a.short_lev:
        ll = a.long_lev or a.short_lev
        scenarios = [(_n(ll), _n(a.short_lev or ll))]
    else:
        scenarios = [(_n(float(x)), _n(float(x))) for x in a.leverages.split(",") if x.strip()]

    summ, curves = run(a.data, [t.strip() for t in a.tfs.split(",")], p, scenarios, a.out)
    with pd.option_context("display.width", 220, "display.max_rows", 1000, "display.max_columns", 50):
        print(summ.to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = summ.dropna(subset=["total_pnl_inr"]).sort_values("total_pnl_inr", ascending=False).head(6)
        fig, ax = plt.subplots(figsize=(10, 5))
        for _, r in top.iterrows():
            s = curves.get((r.symbol, r.tf, r.lev))
            if s is not None:
                ax.plot(pd.to_datetime(s.index), s.values, label=f"{r.symbol} {r.tf} {r.lev}")
        ax.set_ylabel(f"Cumulative P&L (Rs), margin Rs{a.margin:g}/trade")
        ax.set_title("Brahmastra - top equity curves")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "equity_top.png"), dpi=130)
    except Exception as e:
        print("plot skipped:", e)


def _n(x):
    return int(x) if float(x).is_integer() else x


if __name__ == "__main__":
    main()
