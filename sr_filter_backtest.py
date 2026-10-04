"""
S/R filter + backtest (1h timeframe, last 110 candles)

Idea: signal ke time pichle 110 candles ke swing high/low se support/resistance
nikalo. Agar nearest resistance (long) entry se kam se kam 2% door hai, aur
nearest support se SL zyada bada nahi hai, tabhi signal lo. TP = 2% se 3% ke beech.
Short ke liye ulta.

pip install ccxt pandas numpy
"""
import time
import ccxt
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 5000      # backtest history
LOOKBACK = 110               # S/R ke liye last 110 candles
PIVOT_K = 3                  # pivot = left/right 3 candles se high/low
CLUSTER_TOL = 0.004          # 0.4% ke andar ke levels merge
TP_MIN, TP_MAX = 0.02, 0.03  # target 2% se 3%
MAX_SL = 0.015               # max stop loss 1.5%
MAX_HOLD = 48                # max 48 candles (48 hr) tak trade hold
FEE = 0.0005                 # per side fee (0.05%)
# ----------------------------------------


def fetch_data(symbol=SYMBOL, timeframe=TIMEFRAME, total=CANDLES_TO_FETCH):
    ex = ccxt.binance({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - total * tf_ms
    rows = []
    while len(rows) < total:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def get_levels(window: pd.DataFrame, k=PIVOT_K, tol=CLUSTER_TOL):
    """Swing high/low pivots nikalke cluster karta hai. Returns [(level, touches)]."""
    highs, lows = window["high"].values, window["low"].values
    n = len(window)
    pts = []
    for j in range(k, n - k):
        if highs[j] == highs[j - k: j + k + 1].max():
            pts.append(highs[j])
        if lows[j] == lows[j - k: j + k + 1].min():
            pts.append(lows[j])
    pts.sort()

    clusters = []  # [sum, count]
    for p in pts:
        if clusters and abs(p - clusters[-1][0] / clusters[-1][1]) / p < tol:
            clusters[-1][0] += p
            clusters[-1][1] += 1
        else:
            clusters.append([p, 1])
    return [(c[0] / c[1], c[1]) for c in clusters]


def sr_filter(window: pd.DataFrame, entry: float, side: str):
    """
    Returns (ok, tp_price, sl_price).
    Long : nearest resistance entry se >= TP_MIN door ho, nearest support se SL <= MAX_SL.
    Short: nearest support entry se >= TP_MIN neeche ho, nearest resistance se SL <= MAX_SL.
    """
    levels = [l for l, _ in get_levels(window)]
    above = [l for l in levels if l > entry * 1.0005]
    below = [l for l in levels if l < entry * 0.9995]

    if side == "long":
        target_lvl = min(above) if above else None   # nearest resistance
        stop_lvl = max(below) if below else None     # nearest support
        room = (target_lvl - entry) / entry if target_lvl else TP_MAX
        risk = (entry - stop_lvl) / entry if stop_lvl else MAX_SL
        if room < TP_MIN or risk > MAX_SL:
            return False, None, None
        tp = entry * (1 + min(room, TP_MAX))
        sl = stop_lvl * 0.999 if stop_lvl else entry * (1 - MAX_SL)
    else:
        target_lvl = max(below) if below else None   # nearest support
        stop_lvl = min(above) if above else None     # nearest resistance
        room = (entry - target_lvl) / entry if target_lvl else TP_MAX
        risk = (stop_lvl - entry) / entry if stop_lvl else MAX_SL
        if room < TP_MIN or risk > MAX_SL:
            return False, None, None
        tp = entry * (1 - min(room, TP_MAX))
        sl = stop_lvl * 1.001 if stop_lvl else entry * (1 + MAX_SL)
    return True, tp, sl


# ---------------- APNA SIGNAL YAHAN DAALO ----------------
def get_signal(df: pd.DataFrame, i: int):
    """Example: EMA20/EMA50 crossover. Isko apne signal logic se replace karo.
    Return 'long', 'short' ya None. Sirf df.iloc[:i+1] ka data use karna (no lookahead)."""
    if i < 1:
        return None
    f, s = df["ema20"], df["ema50"]
    if f.iloc[i - 1] <= s.iloc[i - 1] and f.iloc[i] > s.iloc[i]:
        return "long"
    if f.iloc[i - 1] >= s.iloc[i - 1] and f.iloc[i] < s.iloc[i]:
        return "short"
    return None
# ---------------------------------------------------------


def simulate(df, i, side, entry, tp, sl):
    """Next candle se trade chalao. Same candle me TP aur SL dono touch ho to SL maana (conservative)."""
    n = len(df)
    for t in range(i + 1, min(i + 1 + MAX_HOLD, n)):
        hi, lo = df["high"].iloc[t], df["low"].iloc[t]
        if side == "long":
            if lo <= sl:
                return "SL", (sl - entry) / entry, t
            if hi >= tp:
                return "TP", (tp - entry) / entry, t
        else:
            if hi >= sl:
                return "SL", (entry - sl) / entry, t
            if lo <= tp:
                return "TP", (entry - tp) / entry, t
    t = min(i + MAX_HOLD, n - 1)
    close = df["close"].iloc[t]
    pnl = (close - entry) / entry if side == "long" else (entry - close) / entry
    return "TIMEOUT", pnl, t


def run_backtest(df, use_filter=True):
    trades = []
    i = LOOKBACK
    while i < len(df) - 1:
        side = get_signal(df, i)
        if side is None:
            i += 1
            continue

        entry = df["close"].iloc[i]  # signal candle close pe entry
        if use_filter:
            window = df.iloc[i - LOOKBACK + 1: i + 1]  # last 110 closed candles
            ok, tp, sl = sr_filter(window, entry, side)
            if not ok:
                i += 1
                continue
        else:  # baseline: fixed 2.5% TP, 1.5% SL
            tp = entry * (1.025 if side == "long" else 0.975)
            sl = entry * (0.985 if side == "long" else 1.015)

        result, pnl, exit_i = simulate(df, i, side, entry, tp, sl)
        trades.append({
            "time": df["time"].iloc[i], "side": side, "entry": entry,
            "tp": tp, "sl": sl, "result": result,
            "pnl_pct": (pnl - 2 * FEE) * 100,
            "bars_held": exit_i - i,
        })
        i = exit_i + 1  # ek time pe ek hi trade
    return pd.DataFrame(trades)


def report(name, tr):
    if tr.empty:
        print(f"\n[{name}] koi trade nahi mila")
        return
    wins = (tr["result"] == "TP").sum()
    print(f"\n===== {name} =====")
    print(f"Trades       : {len(tr)}")
    print(f"TP hit       : {wins} ({wins / len(tr) * 100:.1f}%)")
    print(f"SL hit       : {(tr['result'] == 'SL').sum()}")
    print(f"Timeout      : {(tr['result'] == 'TIMEOUT').sum()}")
    print(f"Avg PnL/trade: {tr['pnl_pct'].mean():.3f}%")
    print(f"Total PnL    : {tr['pnl_pct'].sum():.2f}%")
    print(f"Avg bars held: {tr['bars_held'].mean():.1f}")


if __name__ == "__main__":
    df = fetch_data()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    base = run_backtest(df, use_filter=False)
    filt = run_backtest(df, use_filter=True)
    report("BASELINE (bina S/R filter)", base)
    report("S/R FILTER ke saath", filt)
    filt.to_csv("backtest_trades.csv", index=False)
    print("\nTrades 'backtest_trades.csv' me save ho gaye.")
