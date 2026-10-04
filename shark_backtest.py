"""
SHARK EXCHANGE - S/R FILTER BACKTEST (single file, complete)

Kya karta hai:
  1) Shark Exchange ke public API se 1h candles lata hai (Binance ki zaroorat nahi)
  2) Signal ke time pichle 110 candles ke swing high/low se support/resistance nikalta hai
  2b) SIGNAL = aapka Supertrend flip (1H, closed candle) - get_signal() me
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
LOOKBACK = 110                 # har signal pe S/R ke liye uske pichle 110 candles
BACKTEST_CANDLES = 110         # backtest SIRF last 110 candles (~4.6 din) pe chalega
CANDLES_TO_FETCH = LOOKBACK + BACKTEST_CANDLES + 1   # 221 (1 = abhi chal rahi candle, wo hata di jaati hai)
PIVOT_K = 3                    # swing = left/right 3 candles se high/low
CLUSTER_TOL = 0.004            # 0.4% ke andar ke levels merge
ATR_PERIOD = 14                # aapke code ke hisaab se (message me '10/3' likha tha, par code 14 use karta hai)
ATR_MULTIPLIER = 3.0
MIN_TOUCHES = 2                # level tabhi 'asli' S/R jab kam se kam itni baar touch hua (1 = har pivot)
TP_MIN, TP_MAX = 0.02, 0.03    # target 2% se 3%
MAX_SL = 0.015                 # max stop loss 1.5%
MAX_HOLD = 48                  # max 48 candles hold
FEE = 0.0016                   # per side fee (Shark taker ~0.16%, apne account me verify karo)

QUOTE = "USDT"                 # "USDT" ya "INR" pairs
PRICE_TYPE = "LAST_PRICE"      # ya "MARK_PRICE"
COINS_FILE = "shark_coins.txt"
USE_ALL_COINS = True          # True = Shark ke exchangeInfo se SAARE coins (file ignore). False = shark_coins.txt
ONLY_ALLOWED_COINS = True      # True = sirf neeche wali aapki 334 coins ki list (Shark pe jo available ho). False = Shark ke SAARE coins
ALLOWED_COINS = {x.upper() for x in """
BTC ETH SOL XRP DOGE 1000PEPE 1000SHIB Orca Alch Bless Esp Solv Space 1mbabydoge 1000bonk Bas Manta Wlfi Rsr Xtz Bera
Goat Kaito Cat Pendle Ake Gmt Rave Ray Auction Bluai Xpl Wif Cgpt Meta Meme Pump Lpt Mew Fartcoin Ar Imx Chip Polyx
River Glm Flow Skhynix 1000sats Arc Ava Cetus Gas Sand Apt Ena Crv Linea Arkm Jellyjelly Zrx Evaa Xaut TRX Moca Aster
Hype Kaia Atom Cheems S W Light Xny Mina Link Kite Tst Coin Chillguy Usual Act Awe Cyber Cl Xmr Sapien Avnt Bz VVV Skl
Rez Red People Xag At Cookie Clo Pha Baba Siren Lit Zk Ltc Aztec Wal Folks Inj Flux Jasmy Fil Dym 1000floki Og Op Vana
Lumia Nmr Alt Hbar Aave Ape Uai Pnut Xan Giggle Celo Prom Sei Lab Tao Xau G Velodrome Zora Hmstr Axs Googl Lyn Rare Not
Plume Uni Bard Nil Spk Sqd Tut Spell Xpd Trump Bio Amd Bmt Hana Cys Dram Coti Merl Form Somi Koma Sndk Samsung Xvg Aero
Move Band Pltr Trb Mmt Xlm Moodeng Virtual Vet Lista Pengu Jct Etc Cati Hood Melania Kas Soxl Vtho Grass Natgas Enso
Banana Zen 2z Morpho Rvn Tsla Swarms Render Zec Mon Xpt Arb Dash Skyai Tia Mu B2 Mstr Msft Pyth Koru Xai Myx Intc Irys
Zil Neo Tnsr Mubarak Fet Nvda AMZN Bat Ta Pons Soon Crcl Dood Dexe Anime Marscoin Spcx Bnb Trust Flock Pol Turtle Ens
Dogs Turbo Resolv Avax Sto Cbrs Wld Near H Dot Gala Jup Magic Gua 1000000mog Spx Sahara Zro Cake Strk Theta Comp Akt Cow
Rune Enj Ldo Useless Eigen Ban Allo Paxg Mavia Recall Prompt Bome Tlm Pixel Ada Popcat ICP Syn Mask Saga 1000rats Stbl
Beat Take Esports Met Ethfi Cfx Ondo Fhe Io Pippin Fida Pieverse Bananas31 Wct Snx Sui Tradoor Br Dia Agld Coai Algo 4
Parti Neirocto Egld Santos Brett Ordi M Aevo Kgen Xpin Api3 Drift Yb Apr Ub Hemi Bb Bch Jto Grt The Eul Mito Zerebro
Aixbt Wmt Griffain Ace Safe
""".split()}
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
REQ_GAP = 0.25                 # request ke beech gap (sec). 429 aaye to code khud badha deta hai (max 1.1 = 60 req/min)
MAX_COINS = 0                  # 0 = saare coins. Jaldi test ke liye 20-30 likho
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
    global REQ_GAP
    for _ in range(6):
        _throttle()
        r = requests.request(method, BASE + path, json=body if method == "POST" else None, timeout=20)
        if r.status_code == 429:
            REQ_GAP = min(REQ_GAP * 2, 1.1)   # server ne roka -> dheema karo
            print(f"  429 rate limit, gap ab {REQ_GAP:.2f}s, 10s ruk raha hoon")
            time.sleep(10)
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
        want = min(PAGE_LIMIT, total - len(rows) + 5)      # sirf utne candles maango jitne chahiye
        body = {F_SYMBOL: symbol, F_INTERVAL: interval, F_LIMIT: want,
                F_START: end - want * tf_ms, F_END: end}
        if F_PRICE_TYPE:
            body[F_PRICE_TYPE] = PRICE_TYPE
        batch = _parse(_unwrap(_request("POST", KLINES_PATH, body)))
        if not batch:
            break
        rows += batch
        earliest = min(b[0] for b in batch)
        if earliest >= end or len(batch) < want * 0.5:   # aur data hai hi nahi -> extra request mat karo
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


def _apply_allowed(syms):
    if not ONLY_ALLOWED_COINS:
        return syms
    keep = [x for x in syms if x[: -len(QUOTE)] in ALLOWED_COINS or x.endswith("INR") and x[:-3] in ALLOWED_COINS]
    print(f"ALLOWED_COINS filter: {len(syms)} me se {len(keep)} bache")
    return keep


def get_symbols():
    return _apply_allowed(_get_symbols_raw())


def _get_symbols_raw():
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


# ---------------- AAPKA SUPERTREND SIGNAL ----------------
def supertrend(df, period=ATR_PERIOD, multiplier=ATR_MULTIPLIER):
    """Aapke code ka same Supertrend (RMA-ATR), bas numpy me taaki tez chale.
    direction: 1 = upper band (down trend), -1 = lower band (up trend)."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    pc = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(np.vstack([high - low, np.abs(high - pc), np.abs(low - pc)]), axis=0)

    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / period

    hl2 = (high + low) / 2.0
    upper, lower = hl2 + multiplier * atr, hl2 - multiplier * atr
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    direction, st = np.zeros(n, dtype=int), np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]) or i == period - 1:
            fu[i], fl[i], direction[i], st[i] = upper[i], lower[i], 1, upper[i]
            continue
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] > fu[i] else 1
        else:
            direction[i] = 1 if close[i] < fl[i] else -1
        st[i] = fl[i] if direction[i] == -1 else fu[i]
    return direction, st


def add_indicators(df):
    df["st_dir"], df["st"] = supertrend(df)
    return df


def get_signal(df, i):
    """Aapke scanner jaisa: closed candle i pe Supertrend flip.
    BUY (long)  : prev_close <= prev_ST, close > ST, direction == -1
    SELL (short): prev_close >= prev_ST, close < ST, direction == 1"""
    if i < 2:
        return None
    pc, pst = df["close"].iloc[i - 1], df["st"].iloc[i - 1]
    cc, cst, cd = df["close"].iloc[i], df["st"].iloc[i], df["st_dir"].iloc[i]
    if np.isnan(pst) or np.isnan(cst):
        return None
    if pc <= pst and cc > cst and cd == -1:
        return "long"
    if pc >= pst and cc < cst and cd == 1:
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
    i = max(LOOKBACK, len(df) - BACKTEST_CANDLES)   # sirf last BACKTEST_CANDLES me signal dhoondho
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


def load_df(sym):
    df = fetch_klines(sym)
    tf_ms = TF_SECONDS[TIMEFRAME] * 1000
    if df["ts"].iloc[-1] + tf_ms > time.time() * 1000:   # abhi chal rahi (adhuri) candle hata do
        df = df.iloc[:-1]
    df = df.reset_index(drop=True)
    if len(df) >= LOOKBACK + 30:
        add_indicators(df)
    return df


def scan():
    """Har coin ke SIRF last 110 closed candles lo, abhi ke candle pe signal + S/R filter check karo."""
    symbols = get_symbols()
    if MAX_COINS:
        symbols = symbols[:MAX_COINS]
    tf_ms = TF_SECONDS[TIMEFRAME] * 1000
    print(f"{len(symbols)} coins scan hone hain (Supertrend signal + S/R filter, last candle pe)")
    rows = []
    for n, sym in enumerate(symbols, 1):
        try:
            df = load_df(sym)
            if len(df) < LOOKBACK + 30:
                print(f"[{n}/{len(symbols)}] {sym}: sirf {len(df)} candles mili, skip")
                continue
            i = len(df) - 1
            side = get_signal(df, i)
            if side is None:
                continue
            entry = df["close"].iloc[i]
            ok, tp, sl = sr_filter(df.iloc[i - LOOKBACK + 1: i + 1], entry, side)
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
    if MAX_COINS:
        symbols = symbols[:MAX_COINS]
    print(f"{len(symbols)} coins backtest hone hain (Shark Exchange data)")
    all_base, all_filt = [], []
    fails = 0
    for n, sym in enumerate(symbols, 1):
        if fails >= 3 and not all_filt:
            raise SystemExit("Lagatar 3 coins fail hue. API request format galat lag raha hai. `probe` chalao.")
        try:
            df = load_df(sym)
            if len(df) < LOOKBACK + 30:
                print(f"[{n}/{len(symbols)}] {sym}: data kam hai, skip")
                continue
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
