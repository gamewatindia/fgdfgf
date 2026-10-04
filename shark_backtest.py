"""
SHARK EXCHANGE - S/R FILTER BACKTEST (single file, complete)

Kya karta hai:
  1) Shark Exchange ke public API se 1h candles lata hai (Binance ki zaroorat nahi)
  2) Signal ke time pichle 110 candles ke swing high/low se support/resistance nikalta hai
  3) Long: nearest resistance entry se >= 2% door ho (target 2-3% tak jaa sake) aur
     nearest support se SL <= 1.5% ho, tabhi trade. Short ke liye ulta.
  4) Baseline (bina filter) vs Filter ke saath, dono ka backtest + per-coin report

Chalane ka tarika:
  pip install requests pandas numpy
  python shark_backtest.py probe BTCUSDT     # pehle API check (raw response dikhata hai)
  python shark_backtest.py coins             # Shark se saare coins nikalke shark_coins.txt bana deta hai
  python shark_backtest.py scan              # SIRF last 110 candles lekar abhi ke valid signals dikhata hai
  python shark_backtest.py                   # poora backtest (zyada history ke saath)

Coins: same folder me shark_coins.txt banao (ek line = ek coin, jaise BTC, ETH, SOL).
       Na ho to Shark ke exchangeInfo se list nikalne ki koshish karega.

NOTE: Shark ke klines endpoint ke body-field names docs se confirm nahi ho paaye.
      Agar API error de to `probe` ka output dekho aur neeche "API FIELD NAMES" badlo.
"""
import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd

# ======================= CONFIG =======================
TIMEFRAME = "1h"
CANDLES_TO_FETCH = 3000        # ~125 din ka history
LOOKBACK = 110                 # S/R ke liye last 110 candles
PIVOT_K = 3                    # swing = left/right 3 candles se high/low
CLUSTER_TOL = 0.004            # 0.4% ke andar ke levels merge
MIN_TOUCHES = 2                # level tabhi 'asli' S/R jab kam se kam itni baar touch hua (1 = har pivot)
TP_MIN, TP_MAX = 0.02, 0.03    # target 2% se 3%
MAX_SL = 0.015                 # max stop loss 1.5%
MAX_HOLD = 48                  # max 48 candles hold
FEE = 0.0016                   # per side fee (Shark taker ~0.16%, apne account me verify karo)

QUOTE = "USDT"                 # "USDT" ya "INR" pairs
PRICE_TYPE = "LAST_PRICE"      # ya "MARK_PRICE"
COINS_FILE = "shark_coins.txt"
USE_ALL_COINS = True          # True = Shark ke exchangeInfo se SAARE coins (file ignore). False = shark_coins.txt
CACHE_DIR = "shark_cache"
CACHE_HOURS = 6

# ---- API FIELD NAMES (docs se mismatch ho to yahin badlo) ----
BASE = "https://api.sharkexchange.in"
KLINES_PATH = "/v1/market/klines"
EXCHANGE_INFO_PATH = "/v1/exchange/exchangeInfo"
F_SYMBOL, F_INTERVAL, F_LIMIT = "pair", "interval", "limit"   # API error ne bataya: "symbol" nahi, "pair" chahiye
F_START, F_END = "startTime", "endTime"
F_PRICE_TYPE = None   # API ne "priceType" reject kiya. Baad me sahi naam mile to yahan daalo (None = bhejo mat)
PAGE_LIMIT = 1000
REQ_GAP = 1.1                  # rate limit 60 req/min
# ======================================================

_last_call = [0.0]
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


# ------------------------- SHARK API -------------------------
def _throttle():
    wait = REQ_GAP - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _request(method, path, body=None):
    for _ in range(3):
        _throttle()
        r = requests.request(method, BASE + path, json=body if method == "POST" else None, timeout=20)
        if r.status_code == 429:
            time.sleep(30)
            continue
        if not r.ok:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"{path}: rate limit (429) baar-baar aaya")


def _unwrap(resp):
    if isinstance(resp, dict):
        for k in ("data", "result", "klines", "candles", "contracts", "symbols"):
            if k in resp:
                return resp[k]
    return resp


def _to_ms(v):
    if isinstance(v, str) and not v.replace(".", "", 1).isdigit():
        return int(pd.Timestamp(v).timestamp() * 1000)
    v = float(v)
    return int(v * 1000) if v < 1e11 else int(v)


def _pick(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"{keys} me se koi key nahi mili: {list(d.keys())}")


def _parse(rows):
    out = []
    for r in rows:
        if isinstance(r, (list, tuple)):
            ts, o, h, l, c, v = r[:6]
        else:
            ts = _pick(r, "openTime", "time", "t", "timestamp", "startTime")
            o, h = _pick(r, "open", "o"), _pick(r, "high", "h")
            l, c = _pick(r, "low", "l"), _pick(r, "close", "c")
            v = r.get("volume", r.get("v", 0))
        out.append([_to_ms(ts), float(o), float(h), float(l), float(c), float(v)])
    return out


def fetch_klines(symbol, interval=TIMEFRAME, total=CANDLES_TO_FETCH, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{PRICE_TYPE}_{total}.csv")
    if use_cache and os.path.exists(cache) and time.time() - os.path.getmtime(cache) < CACHE_HOURS * 3600:
        df = pd.read_csv(cache)
        df["time"] = pd.to_datetime(df["ts"], unit="ms")
        return df

    tf_ms = TF_SECONDS[interval] * 1000
    rows, end = [], int(time.time() * 1000)
    while len(rows) < total:
        body = {F_SYMBOL: symbol, F_INTERVAL: interval, F_LIMIT: PAGE_LIMIT,
                F_START: end - PAGE_LIMIT * tf_ms, F_END: end}
        if F_PRICE_TYPE:
            body[F_PRICE_TYPE] = PRICE_TYPE
        batch = _parse(_unwrap(_request("POST", KLINES_PATH, body)))
        if not batch:
            break
        rows += batch
        earliest = min(b[0] for b in batch)
        if earliest >= end:
            break
        end = earliest - 1

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").tail(total).reset_index(drop=True)
    df.to_csv(cache, index=False)
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def _find_symbols(obj, out):
    if isinstance(obj, dict):
        for k in ("symbol", "contractName", "contractPair", "name"):
            if isinstance(obj.get(k), str) and obj[k].isupper():
                out.add(obj[k])
        for v in obj.values():
            _find_symbols(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _find_symbols(v, out)


def symbols_from_api():
    info = None
    for method in ("GET", "POST"):
        try:
            info = _request(method, EXCHANGE_INFO_PATH, {} if method == "POST" else None)
            break
        except Exception as e:
            print(f"exchangeInfo {method} fail: {e}")
    if info is None:
        raise SystemExit("exchangeInfo se list nahi mili. `probe` chalake output bhejo.")
    found = set()
    _find_symbols(info, found)
    return sorted(s for s in found if s.endswith(QUOTE))


def get_symbols():
    if USE_ALL_COINS:
        try:
            syms = symbols_from_api()
            if syms:
                print(f"Shark exchangeInfo se {len(syms)} {QUOTE} coins mile")
                return syms
            print("exchangeInfo se koi symbol nahi nikla, file try kar raha hoon")
        except SystemExit as e:
            print(f"{e} -> file try kar raha hoon")
    if os.path.exists(COINS_FILE):
        with open(COINS_FILE) as f:
            coins = [x.strip().upper() for x in f if x.strip() and not x.strip().startswith("#")]
        return [c if c.endswith(("USDT", "INR")) else c + QUOTE for c in coins]
    syms = symbols_from_api()
    print(f"exchangeInfo se {len(syms)} {QUOTE} symbols mile")
    return syms


def save_coins():
    syms = symbols_from_api()
    with open(COINS_FILE, "w") as f:
        f.write("\n".join(x[: -len(QUOTE)] for x in syms) + "\n")
    print(f"{len(syms)} coins '{COINS_FILE}' me save ho gaye. (Shark pe 334 hone chahiye, count check karo)")


def probe(sym="BTCUSDT"):
    print("== klines raw ==")
    try:
        body = {F_SYMBOL: sym, F_INTERVAL: "1h", F_LIMIT: 5}
        if F_PRICE_TYPE:
            body[F_PRICE_TYPE] = PRICE_TYPE
        raw = _request("POST", KLINES_PATH, body)
        print(json.dumps(raw, indent=1)[:1200])
        print("parsed:", _parse(_unwrap(raw))[:2])
    except Exception as e:
        print("ERROR:", e)
    print("\n== exchangeInfo raw ==")
    for m in ("GET", "POST"):
        try:
            print(m, json.dumps(_request(m, EXCHANGE_INFO_PATH, {} if m == "POST" else None), indent=1)[:1200])
            break
        except Exception as e:
            print(m, "ERROR:", e)


# ------------------------- S/R LOGIC -------------------------
def get_levels(window, k=PIVOT_K, tol=CLUSTER_TOL):
    """Swing high/low pivots ko cluster karke levels deta hai."""
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
    return [c[0] / c[1] for c in clusters if c[1] >= MIN_TOUCHES]


def sr_filter(window, entry, side):
    """Returns (ok, tp_price, sl_price)."""
    levels = get_levels(window)
    above = [l for l in levels if l > entry * 1.0005]
    below = [l for l in levels if l < entry * 0.9995]

    if side == "long":
        tgt = min(above) if above else None      # nearest resistance
        stp = max(below) if below else None      # nearest support
        room = (tgt - entry) / entry if tgt else TP_MAX
        risk = (entry - stp) / entry if stp else MAX_SL
        if room < TP_MIN or risk > MAX_SL:
            return False, None, None
        tp = entry * (1 + min(room, TP_MAX))
        sl = stp * 0.999 if stp else entry * (1 - MAX_SL)
    else:
        tgt = max(below) if below else None      # nearest support
        stp = min(above) if above else None      # nearest resistance
        room = (entry - tgt) / entry if tgt else TP_MAX
        risk = (stp - entry) / entry if stp else MAX_SL
        if room < TP_MIN or risk > MAX_SL:
            return False, None, None
        tp = entry * (1 - min(room, TP_MAX))
        sl = stp * 1.001 if stp else entry * (1 + MAX_SL)
    return True, tp, sl


# ---------------- APNA SIGNAL YAHAN DAALO ----------------
def get_signal(df, i):
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


# ------------------------- BACKTEST -------------------------
def simulate(df, i, side, entry, tp, sl):
    """Same candle me TP aur SL dono touch ho to SL maana (conservative)."""
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
        entry = df["close"].iloc[i]
        if use_filter:
            window = df.iloc[i - LOOKBACK + 1: i + 1]
            ok, tp, sl = sr_filter(window, entry, side)
            if not ok:
                i += 1
                continue
        else:  # baseline: fixed 2.5% TP, 1.5% SL
            tp = entry * (1.025 if side == "long" else 0.975)
            sl = entry * (0.985 if side == "long" else 1.015)
        result, pnl, exit_i = simulate(df, i, side, entry, tp, sl)
        trades.append({"time": df["time"].iloc[i], "side": side, "entry": entry, "tp": tp, "sl": sl,
                       "result": result, "pnl_pct": (pnl - 2 * FEE) * 100, "bars_held": exit_i - i})
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


def scan():
    """Har coin ke SIRF last 110 closed candles lo, abhi ke candle pe signal + S/R filter check karo."""
    symbols = get_symbols()
    tf_ms = TF_SECONDS[TIMEFRAME] * 1000
    print(f"{len(symbols)} coins scan hone hain (har coin ke last {LOOKBACK} candles)")
    rows = []
    for n, sym in enumerate(symbols, 1):
        try:
            df = fetch_klines(sym, TIMEFRAME, LOOKBACK + 1, use_cache=False)
            if df["ts"].iloc[-1] + tf_ms > time.time() * 1000:  # abhi chal rahi candle hata do
                df = df.iloc[:-1]
            df = df.tail(LOOKBACK).reset_index(drop=True)
            if len(df) < LOOKBACK:
                print(f"[{n}/{len(symbols)}] {sym}: sirf {len(df)} candles mili, skip")
                continue
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            i = len(df) - 1
            side = get_signal(df, i)
            if side is None:
                continue
            entry = df["close"].iloc[i]
            ok, tp, sl = sr_filter(df, entry, side)
            if ok:
                rows.append({"symbol": sym, "side": side, "entry": entry, "tp": tp, "sl": sl,
                             "tp_pct": abs(tp - entry) / entry * 100, "sl_pct": abs(sl - entry) / entry * 100,
                             "candle_time": df["time"].iloc[i]})
                print(f"[{n}/{len(symbols)}] SIGNAL {sym} {side} entry={entry:.6g} tp={tp:.6g} sl={sl:.6g}")
        except Exception as e:
            print(f"[{n}/{len(symbols)}] {sym} skip: {e}")
    out = pd.DataFrame(rows)
    if out.empty:
        print("\nAbhi koi valid signal nahi mila.")
    else:
        out.to_csv("scan_signals.csv", index=False)
        print(f"\n{len(out)} valid signals -> scan_signals.csv\n", out.round(4).to_string(index=False))


def main():
    symbols = get_symbols()
    print(f"{len(symbols)} coins backtest hone hain (Shark Exchange data)")
    all_base, all_filt = [], []
    fails = 0
    for n, sym in enumerate(symbols, 1):
        if fails >= 3 and not all_filt:
            raise SystemExit("Lagatar 3 coins fail hue. API request format galat lag raha hai. `probe` chalao.")
        try:
            df = fetch_klines(sym)
            if len(df) < LOOKBACK + 100:
                print(f"[{n}/{len(symbols)}] {sym}: data kam hai, skip")
                continue
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            b, f = run_backtest(df, False), run_backtest(df, True)
            b["symbol"] = f["symbol"] = sym
            all_base.append(b)
            all_filt.append(f)
            fails = 0
            print(f"[{n}/{len(symbols)}] {sym}: base {len(b)} trades, filter {len(f)} trades")
        except Exception as e:
            fails += 1
            print(f"[{n}/{len(symbols)}] {sym} skip: {e}")

    base = pd.concat(all_base, ignore_index=True) if all_base else pd.DataFrame()
    filt = pd.concat(all_filt, ignore_index=True) if all_filt else pd.DataFrame()
    report("BASELINE (bina S/R filter) - saare coins", base)
    report("S/R FILTER ke saath - saare coins", filt)

    if not filt.empty:
        per_coin = filt.groupby("symbol").agg(
            trades=("result", "size"),
            tp_rate=("result", lambda x: (x == "TP").mean() * 100),
            total_pnl=("pnl_pct", "sum"),
        ).sort_values("total_pnl", ascending=False)
        per_coin.to_csv("per_coin_summary.csv")
        filt.to_csv("backtest_trades.csv", index=False)
        print("\nTop 10 coins:\n", per_coin.head(10))
        print("\n'backtest_trades.csv' aur 'per_coin_summary.csv' save ho gayi.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "probe":
        probe(sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT")
    elif cmd == "coins":
        save_coins()
    elif cmd == "scan":
        scan()
    else:
        main()
