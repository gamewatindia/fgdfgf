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

STOP / "NO CHASING" RULES
  --sl 4        stop-loss 4% of price.     --sl pivot   : exit when price breaks the LuxAlgo pivot level itself
                (long: below the pivot low, short: above the pivot high).     --sl pivot,4 : whichever comes first.
  A trade FAILS if it ends by pivot-break / stop-loss / liquidation, or is closed by the opposite pivot at a loss.
  --no_chase    after a failed trade stop trading that coin (single-coin runs).
  Scan string 6th value = rotate:  "40,gate,300,15,24,1"  -> at most 1 coin is traded at a time; a coin whose
                trade FAILED is dropped for the rest of the test (no chasing) and the next coin's pivot is taken.

PUMP / DUMP SCAN (--scan "40,gate,300,15,24")
  Finds coins that moved at least X% inside ONE WINDOW (default 24 hours): PUMP = lowest low -> later high,
  DUMP = highest high -> later low, either one qualifies. Then ALL those coins are backtested over the whole
  timeline and every trade is labelled by when it was entered:
      before  = entered before the first pump/dump started
      during  = entered while that first 24h move was running
      after   = entered after it had already happened
  Results are shown separately for before / during / after (Phase sheets), per timeframe and per first-event type.
  Format: pct[,exchange[,max_coins_to_scan[,max_coins_to_test[,window_hours]]]]   window_hours 0 = old
  whole-period "pump then dump" scan.  --pair ALL = whole exchange, ALL@mexc, A,B,C = only these coins.
  Exchange priority: COIN@exchange in --pair  >  exchange in the scan string  >  default gate (Gate.io).

Every output file name + every row carries run_id / coin / pair / exchange so uploads are self-explaining.
Outputs (folder results/): <run_id>_summary.csv, _trades.csv, _pivots.csv, _scan.csv, _events.csv, _equity.png, <run_id>.xlsx
Run: python pivot_backtest.py --pair BTCUSDT --days 30 --mode hold_long --tp 5 --balance 10000 --max_open 3
Pair can carry the exchange:  --pair BRUSDT@mexc  (ONLY that exchange is used; "gateio" = "gate")
"""
import argparse, os, re, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

MARGIN = 1000.0             # Rs margin per trade
LEV = 10                    # leverage
MMR = 0.005                 # maintenance margin (0.5%)
FEE_PCT = 0.10              # % per side, on full position size
SL = 0.0                    # stop-loss in % of price (0 = off)
SL_PIVOT = False            # also exit when the pivot level breaks
TP = 0.0                    # take-profit in % of price (0 = off)
TFS = ["5m", "15m", "30m", "1h", "4h"]
MODES = ["reverse", "reverse_pivot_price", "hold_long", "hold_short", "hold_both"]
# Tried in order until one has the pair. (bybit/binance.com block GitHub's US servers,
# so they are kept last; gateio/mexc/kucoin/bitget list far more altcoins.)
EXCHANGES = {
    "LAST_PRICE": ["binanceus", "gate", "mexc", "kucoin", "bitget", "okx", "bybit"],
    "MARK_PRICE": ["gate", "bitget", "okx", "bybit", "binanceusdm", "kucoinfutures"],
    "INDEX_PRICE": ["gate", "bitget", "okx", "bybit", "binanceusdm"],
}
# Friendly / old names -> current ccxt id (ccxt renamed gateio -> gate)
ALIASES = {"gateio": "gate", "gate.io": "gate", "gate_io": "gate", "okex": "okx", "huobi": "htx",
           "kucoinfut": "kucoinfutures", "binanceusa": "binanceus"}


def canon(name):
    n = str(name).strip().lower()
    return ALIASES.get(n, n)


def get_exchange(name):
    """name (any alias) -> (ccxt exchange object, ccxt id)."""
    import ccxt
    n = canon(name)
    if not hasattr(ccxt, n):
        raise ValueError(f"exchange '{name}' ccxt me nahi hai")
    return getattr(ccxt, n)({"enableRateLimit": True}), n
HOLD_ONLY_COLS = ["account_liq", "min_equity", "final_equity", "skipped", "skipped_cap", "peak_open"]
ROT_COLS = ["skipped_no_slot", "banned", "ban_time", "ban_reason", "rot_peak_coins"]


def norm_pair(p):
    p = p.upper().replace("-", "/").replace("_", "/")
    if "/" not in p:
        for q in ("USDT", "USDC", "USD"):
            if p.endswith(q):
                return p[: -len(q)] + "/" + q
    return p


def fetch(pair, tf, days, warmup, price, first="", strict=False):
    """Returns (df, info). Tries exchanges in order. Prefers one with the FULL requested history;
    if none has it, uses the one with the most history and prints a warning.
    strict=True (pair written as COIN@exchange): ONLY that exchange is used, no fallback."""
    errs, best = [], None
    first = canon(first) if first else ""
    order = [first] if (strict and first) else \
        ([first] if first else []) + [e for e in EXCHANGES[price] if e != first]
    for name in order:
        try:
            ex, name = get_exchange(name)
            ex.load_markets()
            sym = pair if price == "LAST_PRICE" else pair + ":" + pair.split("/")[1]
            if sym not in ex.markets:
                raise ValueError(f"{sym} not listed")
            fn = getattr(ex, {"LAST_PRICE": "fetch_ohlcv", "MARK_PRICE": "fetch_mark_ohlcv",
                              "INDEX_PRICE": "fetch_index_ohlcv"}[price])
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
            df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
            span = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 86400
            covered = span - warmup * tf_ms / 86400000      # days of history usable for trading
            info = dict(exchange=name, symbol=sym, candles=len(df), days_covered=round(covered),
                        data_from=str(df["time"].iloc[0]), data_to=str(df["time"].iloc[-1]))
            if covered >= days * 0.97:
                print(f"[{tf}] data: {name} {sym} {price} ({len(df)} candles, {covered:.0f} days)")
                return df, info
            errs.append(f"{name}: only {covered:.0f} of {days} days")
            if best is None or covered > best[0]:
                best = (covered, name, sym, df, info)
        except Exception as e:
            errs.append(f"{name}: {str(e)[:60]}")
    if best:
        covered, name, sym, df, info = best
        print(f"[{tf}] WARNING: no exchange had {days} days. Using {name} {sym}: only {covered:.0f} days "
              f"({len(df)} candles). Others: {' | '.join(errs)}")
        return df, info
    raise RuntimeError(" | ".join(errs))


STABLES = {"USDC", "USDT", "DAI", "TUSD", "FDUSD", "BUSD", "USDD", "USDE", "PYUSD", "EUR", "GBP", "USD1"}
PHASES = ["before", "during", "after"]


def parse_scan(s):
    """'0' -> None (off).  pct[,exchange[,max_scan[,max_test[,window_hours[,rotate_slots]]]]]"""
    parts = [p.strip() for p in str(s).split(",")]
    if not parts or parts[0] in ("", "0", "off", "OFF"):
        return None
    pct = float(parts[0])
    if pct <= 0:
        return None
    ex = canon(parts[1]) if len(parts) > 1 and parts[1] else "gate"
    mx_scan = int(parts[2]) if len(parts) > 2 and parts[2] else 300
    mx_test = int(parts[3]) if len(parts) > 3 and parts[3] else 15
    win = int(parts[4]) if len(parts) > 4 and parts[4] else 24
    rot = int(parts[5]) if len(parts) > 5 and parts[5] else 0
    return dict(pct=pct, exchange=ex, max_scan=mx_scan, max_test=mx_test, window=win, rotate=rot)


def fetch_simple(ex, sym, tf, days):
    """Plain candle download from ONE exchange (used by the scanner)."""
    tf_ms = ex.parse_timeframe(tf) * 1000
    now = ex.milliseconds()
    since = now - days * 86400000
    rows = []
    while since < now:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not b:
            break
        rows += b
        nxt = b[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["time", "open", "high", "low", "close"]
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def detect_moves(df, pct, wc):
    """Episodes where price moved >= pct% inside wc candles (e.g. 24 x 1h = 24 hours).
    PUMP = lowest low -> later high, DUMP = highest high -> later low. After an episode is found the same
    direction is not counted again for one more window, so one big move = one event."""
    if len(df) < wc + 5:
        return []
    h, l, t = df["high"].values, df["low"].values, df["time"]
    th = pct / 100
    lo = pd.Series(l).rolling(wc, min_periods=1).min().values      # lowest low in the window ending at j
    hi = pd.Series(h).rolling(wc, min_periods=1).max().values      # highest high in the window ending at j
    events = []
    for typ, mv in (("PUMP", h / lo - 1), ("DUMP", 1 - l / hi)):
        idx = np.flatnonzero(mv >= th)
        p = 0
        while p < len(idx):
            j = idx[p]
            b = j + int(np.argmax(mv[j: j + wc]))                    # strongest point of this episode
            a0 = max(0, b - wc + 1)
            if typ == "PUMP":
                i = a0 + int(np.argmin(l[a0: b + 1])); frm, to = l[i], h[b]
            else:
                i = a0 + int(np.argmax(h[a0: b + 1])); frm, to = h[i], l[b]
            events.append(dict(type=typ, start_time=t.iloc[i], end_time=t.iloc[b],
                               move_pct=round(mv[b] * 100, 1), from_price=frm, to_price=to,
                               pump_pct=round(mv[b] * 100, 1) if typ == "PUMP" else np.nan,
                               dump_pct=round(mv[b] * 100, 1) if typ == "DUMP" else np.nan))
            p = int(np.searchsorted(idx, b + wc))
    events.sort(key=lambda e: e["start_time"])
    return events


def detect_pump_dump(df, pct):
    """Old whole-period scan: pump (lowest low -> peak) AND dump (peak -> lowest low after it), both >= pct."""
    if len(df) < 20:
        return []
    h, l, t = df["high"].values, df["low"].values, df["time"]
    pump = h / np.minimum.accumulate(l) - 1
    suf = np.full(len(l), np.inf)
    suf[:-1] = np.minimum.accumulate(l[::-1])[::-1][1:]          # lowest low strictly after j
    dump = np.where(np.isfinite(suf), 1 - suf / h, 0.0)
    j = int(np.argmax(np.minimum(pump, dump)))
    if min(pump[j], dump[j]) < pct / 100:
        return []
    i = int(np.argmin(l[: j + 1]))
    k = j + 1 + int(np.argmin(l[j + 1:]))
    return [dict(type="PUMP+DUMP", start_time=t.iloc[i], end_time=t.iloc[k],
                 move_pct=round(min(pump[j], dump[j]) * 100, 1), from_price=l[i], to_price=l[k],
                 pump_pct=round(pump[j] * 100, 1), dump_pct=round(dump[j] * 100, 1))]


def scan_pump_dump(pairs, scan, days):
    """Returns (found_df, n_scanned, events_df). pairs=None -> every active USDT spot coin on the exchange."""
    ex, scan["exchange"] = get_exchange(scan["exchange"])
    ex.load_markets()
    tf = "1h" if days <= 120 else "4h"
    wc = max(1, round(scan["window"] / (1 if tf == "1h" else 4))) if scan["window"] else 0
    if pairs:
        syms = [p for p in pairs if p in ex.markets]
        for p in pairs:
            if p not in ex.markets:
                print(f"scan: {p} not listed on {scan['exchange']}")
    else:
        syms = [s for s, m in ex.markets.items()
                if m.get("spot") and m.get("active", True) and m.get("quote") == "USDT"
                and m.get("base") not in STABLES
                and not re.search(r"(\d+[LS]|UP|DOWN|BULL|BEAR)$", str(m.get("base")))]
        try:
            vol = {k: (v.get("quoteVolume") or 0) for k, v in ex.fetch_tickers().items()}
            syms.sort(key=lambda x: -vol.get(x, 0))
        except Exception as e:
            print(f"scan: could not sort by volume ({str(e)[:60]})")
        syms = syms[: scan["max_scan"]]
    what = (f"a >= {scan['pct']:g}% PUMP or DUMP within {scan['window']}h" if wc else
            f"a >= {scan['pct']:g}% pump AND >= {scan['pct']:g}% dump (whole period)")
    print(f"scan: checking {len(syms)} coins on {scan['exchange']} ({tf} candles, last {days} days) for {what}")
    found, evs = [], []
    for n, sym in enumerate(syms, 1):
        try:
            df = fetch_simple(ex, sym, tf, days)
            events = detect_moves(df, scan["pct"], wc) if wc else detect_pump_dump(df, scan["pct"])
        except Exception:
            continue
        if events:
            big = max(events, key=lambda e: e["move_pct"])
            first = events[0]
            pp = [e["pump_pct"] for e in events if e["pump_pct"] == e["pump_pct"]]
            dd = [e["dump_pct"] for e in events if e["dump_pct"] == e["dump_pct"]]
            found.append(dict(pair=sym, coin=sym.split("/")[0], events=len(events),
                              pump_events=sum("PUMP" in e["type"] for e in events),
                              dump_events=sum("DUMP" in e["type"] for e in events),
                              max_pump_pct=max(pp) if pp else "", max_dump_pct=max(dd) if dd else "",
                              biggest_event=big["type"], biggest_move_pct=big["move_pct"],
                              biggest_start=str(big["start_time"]), biggest_end=str(big["end_time"]),
                              first_event=first["type"], first_event_start=str(first["start_time"]),
                              first_event_end=str(first["end_time"]), window_h=scan["window"],
                              scan_tf=tf, scan_exchange=scan["exchange"]))
            evs += [dict(pair=sym, coin=sym.split("/")[0], **e) for e in events]
        if n % 50 == 0:
            print(f"scan: {n}/{len(syms)} checked, {len(found)} found")
    df = pd.DataFrame(found)
    if len(df):
        df = df.sort_values("biggest_move_pct", ascending=False).reset_index(drop=True)
    return df, len(syms), pd.DataFrame(evs)


def add_phase(at, ev_df):
    """Label every trade by entry time vs the coin's FIRST pump/dump: before / during / after."""
    ev = ev_df.copy()
    ev["start_time"], ev["end_time"] = pd.to_datetime(ev["start_time"]), pd.to_datetime(ev["end_time"])
    ev = ev.sort_values("start_time")
    first = ev.groupby("coin").first()
    ends = {c: np.sort(g["end_time"].values) for c, g in ev.groupby("coin")}
    at = at.copy()
    et = pd.to_datetime(at["entry_time"])
    st = at["coin"].map(first["start_time"])
    en = at["coin"].map(first["end_time"])
    at["phase"] = np.where(et < st, "before", np.where(et <= en, "during", "after"))
    at["first_event"] = at["coin"].map(first["type"])
    at["events_done_before_entry"] = [int(np.searchsorted(ends.get(c, np.array([], dtype="datetime64[ns]")),
                                                          np.datetime64(e), side="right"))
                                      for c, e in zip(at["coin"], et)]
    return at


def phase_table(at, by):
    d = at.copy()
    is_open = d["status"] == "open (MTM)"
    d["wins"] = (d["net_pnl"] > 0).astype(int)
    d["realised"] = d["net_pnl"].where(~is_open, 0.0)
    d["open_mtm"] = d["net_pnl"].where(is_open, 0.0)
    d["liquidated"] = d["status"].str.contains("LIQUIDATED").astype(int)
    g = d.groupby(by, sort=False).agg(trades=("net_pnl", "size"), wins=("wins", "sum"),
                                      net_pnl=("net_pnl", "sum"), realised=("realised", "sum"),
                                      open_mtm=("open_mtm", "sum"), liquidated=("liquidated", "sum")).reset_index()
    g["win_pct"] = (g["wins"] / g["trades"] * 100).round(1)
    g["avg_per_trade"] = (g["net_pnl"] / g["trades"]).round(1)
    for c in ("net_pnl", "realised", "open_mtm"):
        g[c] = g[c].round(2)
    g["phase"] = pd.Categorical(g["phase"], PHASES, ordered=True)
    if "tf" in g:
        g["tf"] = pd.Categorical(g["tf"], TFS, ordered=True)
    return g.sort_values(by).reset_index(drop=True).astype({"phase": str, **({"tf": str} if "tf" in g else {})})


def get_signals(df, length, days, at_pivot):
    """{bar_index: (direction, price, pivot_level)}. direction +1 = pivot low (long), -1 = pivot high (short).
    at_pivot=False: signal on the CONFIRMATION bar, price = its close (tradable).
    at_pivot=True : signal on the real pivot bar, price = pivot extreme (look-ahead)."""
    high, low, close, t = df["high"].values, df["low"].values, df["close"].values, df["time"]
    cutoff = t.iloc[-1] - pd.Timedelta(days=days)
    sig = {}
    for i in range(2 * length, len(df)):
        c = i - length
        wh, wl = high[c - length: i + 1], low[c - length: i + 1]
        ph = high[c] == wh.max() and (wh == high[c]).sum() == 1
        pl = low[c] == wl.min() and (wl == low[c]).sum() == 1
        if not (ph or pl):
            continue
        k = c if at_pivot else i
        if t.iloc[k] < cutoff:
            continue
        px = (high[c] if ph else low[c]) if at_pivot else close[i]
        sig[k] = (-1 if ph else 1, px, high[c] if ph else low[c])      # direction, entry price, pivot level
    return sig


def excursion(d, epx, hi, lo):
    """(adverse, favourable) move of this bar vs entry, fraction of price. adverse > 0 = against you."""
    if d == 1:
        return (epx - lo) / epx, (hi - epx) / epx
    return (hi - epx) / epx, (epx - lo) / epx


def parse_sl(s):
    """'0' | '4' | 'pivot' | 'pivot,4'  ->  (percent, use_pivot_level)"""
    pct, piv = 0.0, False
    for tok in re.split(r"[,+ ]+", str(s).strip().lower()):
        if tok in ("", "0", "off"):
            continue
        if tok == "pivot":
            piv = True
        else:
            pct = float(tok)
    return pct, piv


def sl_text():
    bits = ([f"{SL:g}%"] if SL > 0 else []) + (["pivot level"] if SL_PIVOT else [])
    return " + ".join(bits) if bits else "off"


FAIL_STATUSES = ("STOP LOSS", "PIVOT FAILED", "LIQUIDATED")


def is_fail(status, net):
    """A trade FAILED: pivot broken / stop-loss / liquidation, or closed by the opposite pivot at a loss."""
    return status in FAIL_STATUSES or (status == "closed" and net < 0)


def stop_for_trade(d, epx, lvl, liq):
    """(stop distance as fraction of price, which rule) for a new trade; (None, None) if no stop sits
    before liquidation. Pivot rule = price breaking the pivot extreme the entry was based on."""
    cands = []
    if SL > 0:
        cands.append((SL / 100, "STOP LOSS"))
    if SL_PIVOT and lvl is not None:
        dist = (epx - lvl) / epx if d == 1 else (lvl - epx) / epx
        if dist > 0:
            cands.append((dist, "PIVOT FAILED"))
    if not cands:
        return None, None
    dist, kind = min(cands)
    return (dist, kind) if dist < liq else (None, None)


def list_pivots(df, length, days, tf):
    """Every confirmed pivot: when it really happened vs when the strategy could act on it."""
    high, low, t = df["high"].values, df["low"].values, df["time"]
    cutoff = t.iloc[-1] - pd.Timedelta(days=days)
    out = []
    for i in range(2 * length, len(df)):
        c = i - length
        wh, wl = high[c - length: i + 1], low[c - length: i + 1]
        ph = high[c] == wh.max() and (wh == high[c]).sum() == 1
        pl = low[c] == wl.min() and (wl == low[c]).sum() == 1
        if (ph or pl) and t.iloc[c] >= cutoff:
            out.append(dict(tf=tf, type="PIVOT HIGH (short)" if ph else "PIVOT LOW (long)",
                            pivot_time=t.iloc[c], pivot_price=high[c] if ph else low[c],
                            confirmed_time=t.iloc[i], entry_price_if_traded=df["close"].iloc[i]))
    return pd.DataFrame(out)


def reverse_backtest(df, length, days, tf, at_pivot, no_chase=False):
    high, low, t = df["high"].values, df["low"].values, df["time"]
    sig = get_signals(df, length, days, at_pivot)
    notional = MARGIN * LEV
    liq = max(1 / LEV - MMR, 0.001)                  # adverse move that wipes the margin
    tp = TP / 100 if TP > 0 else None
    trades, pos, epx, et, lvl, sd, kind, mae, mfe = [], 0, None, None, None, None, None, 0.0, 0.0

    def close_trade(px, tm, status):
        g = notional * pos * (px / epx - 1)
        fee = notional * FEE_PCT / 100 * 2
        net = max(g - fee, -MARGIN)                  # loss can never exceed the margin
        if status == "LIQUIDATED":
            net = -MARGIN                            # liquidation = full margin lost
        trades.append(dict(tf=tf, side="LONG" if pos == 1 else "SHORT", entry_time=et,
                           entry_price=epx, pivot_price=lvl, exit_time=tm, exit_price=px,
                           mfe_pct=round(mfe * 100, 2), mae_pct=round(mae * 100, 2),
                           hold_h=round((tm - et).total_seconds() / 3600, 1),
                           gross_pnl=round(max(g, -MARGIN), 2), fees=round(fee, 2),
                           net_pnl=round(net, 2), status=status))
        return net

    for j in range(len(df)):
        if pos:
            adv, fav = excursion(pos, epx, high[j], low[j])
            mae, mfe = max(mae, adv), max(mfe, fav)
            status = None
            if sd is not None and adv >= sd:
                status, xpx = kind, epx * (1 - pos * sd)
            elif adv >= liq:
                status, xpx = "LIQUIDATED", epx * (1 - pos * liq)
            elif tp and fav >= tp:
                status, xpx = "TARGET", epx * (1 + pos * tp)
            if status:
                net = close_trade(xpx, t.iloc[j], status); pos = 0
                if no_chase and is_fail(status, net):
                    break                            # failed -> stop trading this coin
        if j in sig:
            d, px, lv = sig[j]
            if d == pos:
                continue
            if pos:
                net = close_trade(px, t.iloc[j], "closed"); pos = 0
                if no_chase and is_fail("closed", net):
                    break                            # opposite pivot at a loss -> exit, do not flip
            pos, epx, et, lvl, mae, mfe = d, px, t.iloc[j], lv, 0.0, 0.0
            sd, kind = stop_for_trade(d, px, lv, liq)
    if pos:
        close_trade(df["close"].iloc[-1], t.iloc[-1], "open (MTM)")
    return pd.DataFrame(trades), len(sig), {}


def rotate_backtest(dfs, order, length, days, tf, at_pivot, slots):
    """Reverse modes on MANY coins with `slots` positions shared between them (time-ordered).
    A coin whose trade FAILS is dropped for the rest of the test (no chasing) and the free slot goes to
    the next coin's pivot signal. `order` = priority when two coins signal on the same candle."""
    notional = MARGIN * LEV
    liq = max(1 / LEV - MMR, 0.001)
    tp = TP / 100 if TP > 0 else None
    sig = {p: get_signals(dfs[p], length, days, at_pivot) for p in order}
    hl = {p: (dfs[p]["high"].values, dfs[p]["low"].values, dfs[p]["time"]) for p in order}
    where = {p: {tm: i for i, tm in enumerate(dfs[p]["time"])} for p in order}
    times = sorted(set().union(*[set(dfs[p]["time"]) for p in order]))
    S = {p: dict(pos=0, epx=None, et=None, lvl=None, sd=None, kind=None, mae=0.0, mfe=0.0) for p in order}
    trades, skipped, banned = {p: [] for p in order}, {p: 0 for p in order}, {}
    open_n = peak = 0

    def close(p, px, tm, status):
        s_ = S[p]
        g = notional * s_["pos"] * (px / s_["epx"] - 1)
        fee = notional * FEE_PCT / 100 * 2
        net = max(g - fee, -MARGIN)
        if status == "LIQUIDATED":
            net = -MARGIN
        trades[p].append(dict(tf=tf, side="LONG" if s_["pos"] == 1 else "SHORT", entry_time=s_["et"],
                              entry_price=s_["epx"], pivot_price=s_["lvl"], exit_time=tm, exit_price=px,
                              mfe_pct=round(s_["mfe"] * 100, 2), mae_pct=round(s_["mae"] * 100, 2),
                              hold_h=round((tm - s_["et"]).total_seconds() / 3600, 1),
                              gross_pnl=round(max(g, -MARGIN), 2), fees=round(fee, 2),
                              net_pnl=round(net, 2), status=status))
        s_["pos"] = 0
        return net

    for tm in times:
        for p in order:
            j = where[p].get(tm)
            if j is None:
                continue
            hi, lo, t = hl[p]
            s_ = S[p]
            if s_["pos"]:
                adv, fav = excursion(s_["pos"], s_["epx"], hi[j], lo[j])
                s_["mae"], s_["mfe"] = max(s_["mae"], adv), max(s_["mfe"], fav)
                status = None
                if s_["sd"] is not None and adv >= s_["sd"]:
                    status, xpx = s_["kind"], s_["epx"] * (1 - s_["pos"] * s_["sd"])
                elif adv >= liq:
                    status, xpx = "LIQUIDATED", s_["epx"] * (1 - s_["pos"] * liq)
                elif tp and fav >= tp:
                    status, xpx = "TARGET", s_["epx"] * (1 + s_["pos"] * tp)
                if status:
                    net = close(p, xpx, t.iloc[j], status); open_n -= 1
                    if is_fail(status, net):
                        banned[p] = (t.iloc[j], status)
            if j in sig[p] and p not in banned:
                d, px, lv = sig[p][j]
                if d == s_["pos"]:
                    continue
                if s_["pos"]:
                    net = close(p, px, t.iloc[j], "closed"); open_n -= 1
                    if is_fail("closed", net):
                        banned[p] = (t.iloc[j], "closed at a loss (opposite pivot)")
                        continue                     # exit and do NOT flip into the opposite side
                if open_n >= slots:
                    skipped[p] += 1                  # all slots busy with other coins
                    continue
                sd_, kind_ = stop_for_trade(d, px, lv, liq)
                s_.update(pos=d, epx=px, et=t.iloc[j], lvl=lv, sd=sd_, kind=kind_, mae=0.0, mfe=0.0)
                open_n += 1
                peak = max(peak, open_n)
    for p in order:
        if S[p]["pos"]:
            close(p, dfs[p]["close"].iloc[-1], dfs[p]["time"].iloc[-1], "open (MTM)")
    out = {p: pd.DataFrame(trades[p]) for p in order}
    extra = {p: dict(skipped_no_slot=skipped[p], banned=p in banned,
                     ban_time=str(banned[p][0]) if p in banned else "",
                     ban_reason=banned[p][1] if p in banned else "", rot_peak_coins=peak) for p in order}
    return out, extra, {p: len(sig[p]) for p in order}


def hold_backtest(df, length, days, tf, side, balance, min_open=0, max_open=0):
    """Cross margin, no SL, opposite signals ignored, every position held until TP.
    max_open: cap on simultaneous positions (0 = no cap).
    min_open: keep at least this many running; a TP that would go below it is postponed."""
    high, low, close, t = df["high"].values, df["low"].values, df["close"].values, df["time"]
    sig = get_signals(df, length, days, False)
    notional = MARGIN * LEV
    fee_side = notional * FEE_PCT / 100
    tp = TP / 100
    cash, opn, trades, skipped, dead, min_eq = balance, [], [], 0, False, balance
    skipped_cap, peak = 0, 0

    def rec(p, px, tm, status):
        g = notional * p["d"] * (px / p["epx"] - 1)
        trades.append(dict(tf=tf, side="LONG" if p["d"] == 1 else "SHORT", entry_time=p["et"],
                           entry_price=p["epx"], exit_time=tm, exit_price=px,
                           mfe_pct=round(p["mfe"] * 100, 2), mae_pct=round(max(p["mae"], 0) * 100, 2),
                           hold_h=round((tm - p["et"]).total_seconds() / 3600, 1),
                           gross_pnl=round(g, 2), fees=round(2 * fee_side, 2),
                           net_pnl=round(g - 2 * fee_side, 2), status=status))
        return g

    for j in range(len(df)):
        if opn:
            worst = 0.0
            for p in opn:
                adv, fav = excursion(p["d"], p["epx"], high[j], low[j])
                p["mae"], p["mfe"], p["fav"] = max(p["mae"], adv), max(p["mfe"], fav), fav
                worst -= adv * notional              # unrealised P&L at this bar's worst price
            eq = cash + worst
            min_eq = min(min_eq, eq)
            if eq <= len(opn) * notional * MMR:      # cross-margin: whole account liquidated
                for p in opn:
                    rec(p, low[j] if p["d"] == 1 else high[j], t.iloc[j], "ACCOUNT LIQUIDATED")
                opn, cash, dead = [], 0.0, True
                break
            for p in list(opn):
                hit = p["fav"] >= tp
                if hit or p.get("due"):
                    if len(opn) <= min_open:         # keep the minimum number of trades running
                        p["due"] = True
                        continue
                    if hit:
                        g = rec(p, p["epx"] * (1 + p["d"] * tp), t.iloc[j], "TARGET")
                    else:                            # postponed TP: exit at market once allowed
                        g = rec(p, close[j], t.iloc[j], "DEFERRED EXIT (min_open)")
                    cash += g - fee_side
                    opn.remove(p)
        if j in sig:
            d, px, _lvl = sig[j]
            if side and d != side:
                continue
            if max_open and len(opn) >= max_open:    # simultaneous-trade cap reached
                skipped_cap += 1
                continue
            unreal = sum(notional * p["d"] * (close[j] / p["epx"] - 1) for p in opn)
            if cash + unreal - len(opn) * MARGIN < MARGIN + fee_side:   # no free margin left
                skipped += 1
                continue
            cash -= fee_side
            opn.append(dict(d=d, epx=px, et=t.iloc[j], mae=0.0, mfe=0.0, fav=0.0))
            peak = max(peak, len(opn))
    unreal = 0.0
    for p in opn:
        rec(p, close[-1], t.iloc[-1], "open (MTM)")
        unreal += notional * p["d"] * (close[-1] / p["epx"] - 1)
    extra = dict(account_liq=dead, min_equity=round(min_eq if not dead else 0.0, 2),
                 final_equity=round(cash + unreal, 2), skipped=skipped, skipped_cap=skipped_cap,
                 peak_open=peak)
    return pd.DataFrame(trades), len(sig), extra


def summarize(tf, tr, npv, extra=None, err=""):
    s = dict(tf=tf, pivots=npv, trades=0, long=0, short=0, wins=0, win_pct=0.0, net_pnl=0.0,
             realised=0.0, open_mtm=0.0, tp_hits=0, open_n=0, liq=0, sl_hits=0, worst_mae=0.0,
             fees=0.0, profit_factor=0.0, max_dd=0.0, account_liq="", min_equity="",
             final_equity="", skipped="", skipped_cap="", peak_open="", skipped_no_slot="", banned="",
             ban_time="", ban_reason="", rot_peak_coins="", note=err)
    s.update(extra or {})
    if tr.empty:
        return s
    w, l = tr[tr.net_pnl > 0], tr[tr.net_pnl <= 0]
    eq = pd.concat([pd.Series([0.0]), tr.net_pnl.cumsum()], ignore_index=True)
    gl = abs(l.net_pnl.sum())
    s.update(trades=len(tr), long=int((tr.side == "LONG").sum()), short=int((tr.side == "SHORT").sum()),
             wins=len(w), win_pct=round(len(w) / len(tr) * 100, 1), net_pnl=round(tr.net_pnl.sum(), 2),
             realised=round(tr[tr.status != "open (MTM)"].net_pnl.sum(), 2),
             open_mtm=round(tr[tr.status == "open (MTM)"].net_pnl.sum(), 2),
             tp_hits=int((tr.status == "TARGET").sum()),
             open_n=int((tr.status == "open (MTM)").sum()),
             liq=int(tr.status.str.contains("LIQUIDATED").sum()),
             sl_hits=int((tr.status == "STOP LOSS").sum()),
             worst_mae=round(tr.mae_pct.max(), 2), fees=round(tr.fees.sum(), 2),
             profit_factor=round(w.net_pnl.sum() / gl, 2) if gl else float("inf"),
             max_dd=round((eq - eq.cummax()).min(), 2))
    return s


def plot_all(all_tr, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for tf, g in all_tr.groupby("tf", sort=False):
        g = g.sort_values("exit_time")
        ax.step(g.exit_time, g.net_pnl.cumsum(), where="post", marker="o", label=tf)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(title + "\nEquity curves (Rs, net of fees)", fontsize=10)
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=120)


LEGEND = [
    ("run_id", "Unique name of this run: coin(or SCAN)-mode-days-leverage-(tp/sl/limits)-UTC time"),
    ("coin / pair / exchange", "Coin, trading pair, and the exchange the candles came from"),
    ("mode", "reverse | reverse_pivot_price (hindsight, NOT tradable) | hold_long | hold_short | hold_both"),
    ("tf", "Candle timeframe of that row"),
    ("pivots / trades / long / short", "Confirmed pivots in window; trades taken; split by side"),
    ("net_pnl", "Rs profit/loss after fees. Includes open trades marked at the last price"),
    ("realised / open_mtm", "Closed-trade P&L / unrealised P&L of trades still open at the end"),
    ("tp_hits / open_n", "Trades that hit take-profit / trades still open at the end"),
    ("liq / sl_hits", "Liquidations (trade or whole account) / stop-loss exits"),
    ("worst_mae", "Worst move AGAINST a trade, in % of price (10x: 10% = 100% of margin)"),
    ("max_dd", "Worst fall of cumulative closed P&L from its peak, Rs"),
    ("account_liq / min_equity / final_equity", "Hold modes: whole account liquidated? lowest and final equity, Rs"),
    ("peak_open", "Hold modes: highest number of trades open at the same time"),
    ("skipped / skipped_cap", "Hold modes: signals skipped for no free margin / for max_open limit"),
    ("status (trades)", "TARGET, closed (opposite pivot), STOP LOSS, PIVOT FAILED (price broke the pivot level), "
                        "LIQUIDATED, ACCOUNT LIQUIDATED, DEFERRED EXIT (min_open), open (MTM)"),
    ("pivot_price", "The LuxAlgo pivot extreme the trade was based on (pivot low for longs, pivot high for shorts)"),
    ("FAILED trade", "PIVOT FAILED, STOP LOSS, LIQUIDATED, or closed by the opposite pivot at a loss"),
    ("skipped_no_slot / banned / ban_time / ban_reason", "Rotation runs: signals skipped because all slots were "
                        "busy / coin dropped after a failed trade (no chasing), when and why"),
    ("Rotation total / Dropped coins sheets", "Per timeframe: coins traded, coins dropped after a fail, trades, P&L; "
                                              "and the list of dropped coins"),
    ("mfe_pct / mae_pct / hold_h", "Best move for / worst move against the trade (% of price); hours held"),
    ("min_open / max_open", "Hold modes: minimum trades kept running / maximum trades at the same time (0 = off)"),
    ("Scan sheet", "One row per coin that had a pump or dump >= scan % inside the window (default 24h). "
                   "events = how many such moves, biggest_* = the largest one, first_event_* = the earliest one"),
    ("Events sheet", "Every single pump/dump episode: type PUMP (low->high) or DUMP (high->low), start/end time, "
                     "move_pct, from/to price"),
    ("phase (Trades sheet)", "before = trade entered before the coin's first pump/dump started; during = entered "
                             "while it was running; after = entered after it had already happened"),
    ("events_done_before_entry", "How many pump/dump episodes of that coin had already finished when the trade opened"),
    ("Phase total / by coin / by type", "Results split by phase (before/during/after): trades, wins, net_pnl, "
                                        "realised, open_mtm, liquidated, win_pct, avg_per_trade"),
    ("Coin x TF sheet", "Net P&L per coin and timeframe (scan runs only)"),
]


def write_excel(path, summary, trades, pivots, settings, extra=None):
    try:
        import openpyxl
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("openpyxl not installed -> Excel file skipped (CSV files are still saved)")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        (trades if trades is not None and len(trades) else pd.DataFrame({"info": ["no trades"]})) \
            .to_excel(xw, sheet_name="Trades", index=False)
        (pivots if pivots is not None and len(pivots) else pd.DataFrame({"info": ["no pivots"]})) \
            .to_excel(xw, sheet_name="Pivots", index=False)
        for name, d in (extra or {}).items():
            d.to_excel(xw, sheet_name=name, index=False)
        pd.DataFrame(list(settings.items()), columns=["setting", "value"]) \
            .to_excel(xw, sheet_name="Settings", index=False)
        pd.DataFrame(LEGEND, columns=["column", "meaning"]).to_excel(xw, sheet_name="Legend", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(ws.columns, 1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:300])
                ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 8), 45)


def gh_summary(sm, args, label, run_id, scan, scan_df, n_scanned, phase_tot=None, phase_type=None, rot_tot=None):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    hold = args.mode.startswith("hold")
    with open(path, "a") as f:
        f.write(f"## {label} | {args.mode} | {args.days}d | {LEV:g}x | SL {sl_text()} | TP {TP:g}%"
                f"{f' | open {args.min_open}-{args.max_open}' if hold else ''} | pivot length {args.length}\n\n")
        f.write(f"`run_id: {run_id}`\n\n")
        if scan:
            n = 0 if scan_df is None else len(scan_df)
            kind = (f">= {scan['pct']:g}% PUMP or DUMP within {scan['window']}h" if scan["window"] else
                    f">= {scan['pct']:g}% pump AND dump (whole period)")
            f.write(f"**Scan** ({scan['exchange']}, {kind}): {n_scanned} coins checked, **{n} found**, "
                    f"{min(n, scan['max_test'])} tested\n\n")
            if n:
                f.write("| Coin | Events (pump/dump) | Biggest | Move % | When | First event |\n|---|---|---|---|---|---|\n")
                for r in scan_df.head(scan["max_test"]).itertuples():
                    f.write(f"| {r.coin} | {r.events} ({r.pump_events}/{r.dump_events}) | {r.biggest_event} | "
                            f"{r.biggest_move_pct} | {r.biggest_start[:16]} to {r.biggest_end[:16]} | "
                            f"{r.first_event} |\n")
                f.write("\n")
        if args.mode == "reverse_pivot_price":
            f.write("> NOT TRADABLE: entry at the real pivot price needs future data. Comparison only.\n\n")
        if sm is None or not len(sm):
            f.write("No coins to test.\n")
            return
        if rot_tot is not None and len(rot_tot):
            f.write(f"**Rotation: {scan['rotate']} slot(s) shared by the coins. A coin whose trade FAILS is dropped "
                    "(no chasing) and the next coin's pivot is taken**\n\n")
            f.write("| TF | Coins | Traded | Dropped after fail | Trades | Net P&L | Realised | Open (MTM) | "
                    "Liquidated | Pivot failed | Stop-loss | Skipped (no slot) | Max coins open |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in rot_tot.itertuples():
                f.write(f"| {r.tf} | {r.coins} | {r.coins_traded} | {r.coins_dropped_after_fail} | {r.trades} | "
                        f"{r.net_pnl} | {r.realised} | {r.open_mtm} | {r.liquidated} | {r.pivot_failed} | "
                        f"{r.stop_loss} | {r.skipped_no_slot} | {r.max_coins_open} |\n")
            f.write("\n")
        if phase_tot is not None and len(phase_tot):
            f.write("**Trades split by WHEN they were entered** (before / during / after the coin's first "
                    "pump or dump), all coins together\n\n")
            f.write("| TF | Phase | Trades | Win% | Net P&L | Realised | Open (MTM) | Liquidated |\n"
                    "|---|---|---|---|---|---|---|---|\n")
            for r in phase_tot.itertuples():
                f.write(f"| {r.tf} | {r.phase} | {r.trades} | {r.win_pct}% | {r.net_pnl} | {r.realised} | "
                        f"{r.open_mtm} | {r.liquidated} |\n")
            f.write("\n")
        if phase_type is not None and len(phase_type):
            f.write("**Same split by type of the first event** (all timeframes together)\n\n")
            f.write("| First event | Phase | Trades | Win% | Net P&L | Liquidated |\n|---|---|---|---|---|---|\n")
            for r in phase_type.itertuples():
                f.write(f"| {r.first_event} | {r.phase} | {r.trades} | {r.win_pct}% | {r.net_pnl} | "
                        f"{r.liquidated} |\n")
            f.write("\n")
        f.write("| Coin | TF | Exchange | Trades | TP hits | Still open | Win% | Net P&L | Realised | Open (MTM) | "
                "Liquidated | SL hits | Worst adverse % | Fees | PF | Max DD |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in sm.itertuples():
            f.write(f"| {r.coin} | {r.tf} | {r.exchange} | {r.trades} | {r.tp_hits} | {r.open_n} | {r.win_pct}% | "
                    f"{r.net_pnl} | {r.realised} | {r.open_mtm} | {r.liq} | {r.sl_hits} | {r.worst_mae} | "
                    f"{r.fees} | {r.profit_factor} | {r.max_dd} |{' ' + r.note if r.note else ''}\n")
        if hold:
            f.write(f"\n**Cross-margin account (start Rs {args.balance:g} per coin)**\n\n")
            f.write("| Coin | TF | Account liquidated? | Lowest equity | Final equity | Peak open trades | "
                    "Skipped (no margin) | Skipped (max_open) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in sm.itertuples():
                f.write(f"| {r.coin} | {r.tf} | {r.account_liq} | {r.min_equity} | {r.final_equity} | "
                        f"{r.peak_open} | {r.skipped} | {r.skipped_cap} |\n")
        f.write(f"\nMargin Rs {MARGIN:.0f} x {LEV:g}x = Rs {MARGIN*LEV:.0f} position, fee {FEE_PCT}%/side "
                "on position. Funding & slippage ignored. Few trades = low statistical value.\n")


def tag_df(d, run_id, pair, mode, ex_by_tf):
    """Put run_id / coin / pair / exchange / mode in front of every row."""
    d = d.copy()
    d.insert(0, "run_id", run_id); d.insert(1, "coin", pair.split("/")[0]); d.insert(2, "pair", pair)
    d.insert(3, "exchange", d["tf"].map(ex_by_tf)); d.insert(4, "mode", mode)
    return d


def meta_row(row, run_id, pair, a, hold, info, ts):
    row.update(run_id=run_id, coin=pair.split("/")[0], pair=pair, mode=a.mode, exchange=info.get("exchange", ""),
               price_type=a.price, days=a.days, leverage=LEV, margin_rs=MARGIN, sl_pct=SL, sl_pivot=SL_PIVOT,
               tp_pct=TP, balance_rs=a.balance if hold else "", min_open=a.min_open if hold else "",
               max_open=a.max_open if hold else "", pivot_length=a.length, fee_pct=FEE_PCT,
               candles=info.get("candles", ""), days_covered=info.get("days_covered", ""),
               data_from=info.get("data_from", ""), data_to=info.get("data_to", ""),
               run_time_utc=ts.strftime("%Y-%m-%d %H:%M"))
    return row


def run_coin(pair, a, first, run_id, ts, hold, strict=False):
    """Backtest ONE coin on all timeframes -> (summary rows, trades df|None, pivots df|None)."""
    rows, trs, pvs, ex_by_tf = [], [], [], {}
    for tf in a.tfs.split(","):
        info = {}
        try:
            df, info = fetch(pair, tf, a.days, 2 * a.length + 5, a.price, first, strict)
            ex_by_tf[tf] = info["exchange"]
            pvs.append(list_pivots(df, a.length, a.days, tf))
            if hold:
                side = {"hold_long": 1, "hold_short": -1, "hold_both": 0}[a.mode]
                tr, npv, extra = hold_backtest(df, a.length, a.days, tf, side, a.balance, a.min_open, a.max_open)
            else:
                tr, npv, extra = reverse_backtest(df, a.length, a.days, tf, a.mode == "reverse_pivot_price",
                                                  a.no_chase)
            row = summarize(tf, tr, npv, extra)
            if not tr.empty:
                trs.append(tr)
        except Exception as e:
            print(f"[{pair} {tf}] ERROR: {e}")
            row = summarize(tf, pd.DataFrame(), 0, None, "ERROR: " + str(e)[:300])
        rows.append(meta_row(row, run_id, pair, a, hold, info, ts))
    trades = tag_df(pd.concat(trs, ignore_index=True), run_id, pair, a.mode, ex_by_tf) if trs else None
    pivots = tag_df(pd.concat(pvs, ignore_index=True), run_id, pair, a.mode, ex_by_tf) \
        if pvs and any(len(p) for p in pvs) else None
    return rows, trades, pivots


def run_rotation(coins, a, first, run_id, ts, slots):
    """Reverse modes, many coins: shared slots; a coin whose trade fails is dropped (no chasing)."""
    rows, trs, pvs = [], [], []
    ex_map = {p: {} for p in coins}
    for tf in a.tfs.split(","):
        dfs, infos, order = {}, {}, []
        for pr in coins:
            try:
                df, info = fetch(pr, tf, a.days, 2 * a.length + 5, a.price, first, True)
                dfs[pr], infos[pr] = df, info
                order.append(pr)
                ex_map[pr][tf] = info["exchange"]
                pv = list_pivots(df, a.length, a.days, tf)
                if len(pv):
                    pvs.append(tag_df(pv, run_id, pr, a.mode, ex_map[pr]))
            except Exception as e:
                print(f"[{pr} {tf}] ERROR: {e}")
                row = summarize(tf, pd.DataFrame(), 0, None, "ERROR: " + str(e)[:300])
                rows.append(meta_row(row, run_id, pr, a, False, {}, ts))
        if not order:
            continue
        res, extra, npv = rotate_backtest(dfs, order, a.length, a.days, tf, a.mode == "reverse_pivot_price", slots)
        dropped = [p.split("/")[0] for p in order if extra[p]["banned"]]
        print(f"[{tf}] rotation: {len(order)} coins, {slots} slot(s) -> dropped after a failed trade: "
              f"{', '.join(dropped) if dropped else 'none'}")
        for pr in order:
            tr = res[pr]
            rows.append(meta_row(summarize(tf, tr, npv[pr], extra[pr]), run_id, pr, a, False, infos[pr], ts))
            if not tr.empty:
                trs.append(tag_df(tr, run_id, pr, a.mode, ex_map[pr]))
    trades = pd.concat(trs, ignore_index=True) if trs else None
    pivots = pd.concat(pvs, ignore_index=True) if pvs else None
    return rows, trades, pivots


def rotation_table(sm, at):
    d = sm.copy()
    d["banned_n"] = d["banned"].map(lambda v: 1 if v is True else 0)
    d["traded"] = (d["trades"] > 0).astype(int)
    for c in ("skipped_no_slot", "rot_peak_coins"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    g = d.groupby("tf", sort=False).agg(
        coins=("coin", "nunique"), coins_traded=("traded", "sum"), coins_dropped_after_fail=("banned_n", "sum"),
        trades=("trades", "sum"), net_pnl=("net_pnl", "sum"), realised=("realised", "sum"),
        open_mtm=("open_mtm", "sum"), liquidated=("liq", "sum"), skipped_no_slot=("skipped_no_slot", "sum"),
        max_coins_open=("rot_peak_coins", "max")).reset_index()
    for st, col in (("PIVOT FAILED", "pivot_failed"), ("STOP LOSS", "stop_loss")):
        cnt = at[at["status"] == st].groupby("tf").size() if at is not None else {}
        g[col] = g["tf"].map(cnt).fillna(0).astype(int)
    for c in ("net_pnl", "realised", "open_mtm"):
        g[c] = g[c].round(2)
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BTCUSDT",
                    help="BTCUSDT | BRUSDT@mexc (only that exchange) | with --scan: ALL, ALL@mexc or A,B,C")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--price", default="LAST_PRICE", choices=list(EXCHANGES))
    ap.add_argument("--length", type=int, default=50)
    ap.add_argument("--tfs", default=",".join(TFS))
    ap.add_argument("--first", default="", help="preferred exchange tried first, others as fallback")
    ap.add_argument("--lev", type=float, default=10)
    ap.add_argument("--margin", type=float, default=1000)
    ap.add_argument("--sl", default="0", help='stop-loss: "4" = 4%% of price, "pivot" = exit when the pivot level '
                    'breaks, "pivot,4" = whichever first, "0" = off')
    ap.add_argument("--no_chase", action="store_true", help="after a failed trade stop trading that coin")
    ap.add_argument("--tp", type=float, default=0, help="take-profit %% of price, 0 = off")
    ap.add_argument("--mode", default="reverse", choices=MODES)
    ap.add_argument("--balance", type=float, default=10000, help="cross-margin account balance (hold modes)")
    ap.add_argument("--min_open", type=int, default=0, help="hold modes: keep at least N trades running (0 = off)")
    ap.add_argument("--max_open", type=int, default=0, help="hold modes: at most N trades at once (0 = no cap)")
    ap.add_argument("--scan", default="0", help='pump&dump scan: "0" = off, or "40,gateio,300,15" = '
                    "pct,exchange,max_scan,max_test")
    ap.add_argument("--out", default="results", help="output folder")
    a = ap.parse_args()
    LEV, MARGIN, TP = a.lev, a.margin, a.tp
    SL, SL_PIVOT = parse_sl(a.sl)
    hold = a.mode.startswith("hold")
    if a.min_open < 0 or a.max_open < 0:
        ap.error("min_open / max_open cannot be negative")
    if a.max_open and a.min_open >= a.max_open:
        ap.error("min_open must be smaller than max_open (else no trade could ever close)")
    try:
        scan = parse_scan(a.scan)
    except ValueError:
        ap.error('scan must look like "0" (off) or "40" or "40,gateio,300,15"')
    rot = scan["rotate"] if scan else 0
    if rot and hold:
        print("note: rotation (shared slots) applies to reverse modes only -> ignored")
        rot = 0
    if hold and TP <= 0:
        TP = 5.0
        print("hold mode needs a target -> using TP = 5% of price")
    if not hold and (a.min_open or a.max_open):
        print("note: min_open / max_open apply to hold modes only (reverse modes hold 1 trade at a time)")
    ex_pref, items = "", []
    for item in a.pair.strip().split(","):          # BTCUSDT@mexc  or  A@mexc,B,C
        if "@" in item:
            item, e = item.split("@", 1)
            ex_pref = ex_pref or canon(e)
        items.append(item.strip())
    raw_pair = ",".join(i for i in items if i)
    first = ex_pref or canon(a.first)                # exchange written next to the coin wins
    strict = bool(ex_pref)                           # COIN@exchange = use ONLY that exchange
    ts = datetime.now(timezone.utc)

    scan_df, n_scanned, ev_df = None, 0, None
    if scan:
        if ex_pref:
            scan["exchange"] = ex_pref                 # pair@exchange beats the scan string / default
        scan["exchange"] = canon(scan["exchange"])
        first, strict = scan["exchange"], True         # test the coins on the exchange they were found on
        pairs = None if raw_pair.upper() in ("", "ALL", "*") else \
            [norm_pair(p.strip()) for p in raw_pair.split(",") if p.strip()]
        scan_df, n_scanned, ev_df = scan_pump_dump(pairs, scan, a.days)
        coins = list(scan_df["pair"][: scan["max_test"]]) if len(scan_df) else []
        label = f"SCAN{scan['pct']:g}pct{str(scan['window']) + 'h' if scan['window'] else 'PD'}-{len(coins)}coins"
        print(f"\nscan result: {len(scan_df)} coins qualify -> testing {len(coins)}: "
              + ", ".join(c.split('/')[0] for c in coins))
    else:
        pair = norm_pair(raw_pair)
        coins = [pair]
        label = pair.split("/")[0]

    parts = [label, a.mode, f"{a.days}d", f"{LEV:g}x"]
    if TP > 0:
        parts.append(f"tp{TP:g}")
    if SL > 0:
        parts.append(f"sl{SL:g}")
    if SL_PIVOT:
        parts.append("slpivot")
    if rot:
        parts.append(f"rot{rot}")
    elif a.no_chase:
        parts.append("nochase")
    if hold and (a.min_open or a.max_open):
        parts.append(f"open{a.min_open}-{a.max_open}")
    run_id = "-".join(parts) + "-" + ts.strftime("%Y%m%d_%H%M")
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, run_id)

    all_rows, all_tr, all_pv = [], [], []
    if rot and coins:
        print(f"\nROTATION: {rot} slot(s) shared by {len(coins)} coin(s); a coin whose trade FAILS is dropped "
              f"for the rest of the test (no chasing) and the next coin's pivot is taken")
        rows, trades, pivots = run_rotation(coins, a, first, run_id, ts, rot)
        all_rows += rows
        if trades is not None:
            all_tr.append(trades)
        if pivots is not None:
            all_pv.append(pivots)
    else:
        for n, pr in enumerate(coins, 1):
            if scan:
                r0 = scan_df.iloc[n - 1]
                print(f"\n===== coin {n}/{len(coins)}: {pr}  ({r0.events} event(s), biggest {r0.biggest_event} "
                      f"{r0.biggest_move_pct}%) =====")
            rows, trades, pivots = run_coin(pr, a, first, run_id, ts, hold, strict)
            all_rows += rows
            if trades is not None:
                all_tr.append(trades)
            if pivots is not None:
                all_pv.append(pivots)

    sm = pd.DataFrame(all_rows)
    if len(sm):
        lead = ["run_id", "coin", "pair", "mode", "tf", "exchange"]
        sm = sm[lead + [c for c in sm.columns if c not in lead + ["note"]] + ["note"]]
        drop = ([] if hold else HOLD_ONLY_COLS + ["balance_rs", "min_open", "max_open"]) + ([] if rot else ROT_COLS)
        smd = sm.drop(columns=drop) if drop else sm
    else:
        smd = pd.DataFrame({"info": ["no coins qualified for the pump & dump scan - nothing tested"]})
    at = pv = None
    if all_tr:
        at = pd.concat(all_tr, ignore_index=True)
        at.insert(at.columns.get_loc("tf") + 1, "trade_no", at.groupby(["coin", "tf"]).cumcount() + 1)
    if all_pv:
        pv = pd.concat(all_pv, ignore_index=True)
    phase_tot = phase_coin = phase_type = phase_type_gh = None
    if scan and at is not None and ev_df is not None and len(ev_df):
        at = add_phase(at, ev_df[ev_df["pair"].isin(coins)])
        phase_tot = phase_table(at, ["tf", "phase"])
        phase_coin = phase_table(at, ["coin", "tf", "phase"])
        phase_type = phase_table(at, ["first_event", "tf", "phase"])
        phase_type_gh = phase_table(at, ["first_event", "phase"])

    print("\n" + "=" * 78)
    print(run_id)
    print(f"{label} | {a.price} | {a.days}d | mode {a.mode} | length {a.length} | margin Rs {MARGIN:.0f} x {LEV:g}x"
          f" | SL {sl_text()} | TP {TP:g}% | fee {FEE_PCT}%/side"
          + (f" | balance Rs {a.balance:g} | open {a.min_open}-{a.max_open}" if hold else ""))
    if a.mode == "reverse_pivot_price":
        print("NOTE: entry at real pivot price is NOT tradable (look-ahead) - comparison only.")
    print("=" * 78)
    if len(sm):
        show = ["coin", "tf", "exchange", "trades", "long", "short", "win_pct", "net_pnl", "realised", "open_mtm",
                "tp_hits", "open_n", "liq", "sl_hits", "worst_mae", "fees", "profit_factor", "max_dd"]
        if hold:
            show += ["account_liq", "min_equity", "final_equity", "peak_open", "skipped", "skipped_cap"]
        if rot:
            show += ["skipped_no_slot", "banned", "ban_time", "ban_reason"]
        print(smd[show].to_string(index=False))
    else:
        print(smd.iloc[0, 0])
    smd.to_csv(base + "_summary.csv", index=False)
    if scan_df is not None:
        scan_df.to_csv(base + "_scan.csv", index=False)
        if len(scan_df):
            cols = ["coin", "events", "pump_events", "dump_events", "max_pump_pct", "max_dump_pct",
                    "biggest_event", "biggest_move_pct", "biggest_start", "first_event"]
            print("\nSCAN RESULT\n" + scan_df[cols].to_string(index=False))
    if ev_df is not None and len(ev_df):
        ev_df.to_csv(base + "_events.csv", index=False)
    if phase_tot is not None:
        print("\nTRADES BY ENTRY PHASE (before / during / after the coin's first pump or dump)\n"
              + phase_tot.drop(columns=["wins", "avg_per_trade"]).to_string(index=False))
    if pv is not None:
        pv.to_csv(base + "_pivots.csv", index=False)
    if at is not None:
        at.to_csv(base + "_trades.csv", index=False)
        if not scan:
            print("\nALL TRADES - last 150 (full list in file)\n"
                  + at.tail(150).drop(columns=["run_id", "coin", "pair", "mode"]).to_string(index=False))
        plot_all(at, base + "_equity.png", f"{label} | {a.mode} | {a.days}d | {LEV:g}x"
                 + (f" | TP {TP:g}%" if TP else "") + (f" | SL {sl_text()}" if (SL or SL_PIVOT) else "")
                 + (f" | rotate {rot}" if rot else ""))
    extra = {}
    if scan_df is not None:
        extra["Scan"] = scan_df if len(scan_df) else pd.DataFrame({"info": ["no coin qualified"]})
    if ev_df is not None and len(ev_df):
        extra["Events"] = ev_df
    if phase_tot is not None:
        extra["Phase total"], extra["Phase by coin"], extra["Phase by type"] = phase_tot, phase_coin, phase_type
    rot_tot = None
    if rot and len(sm):
        rot_tot = rotation_table(sm, at)
        extra["Rotation total"] = rot_tot
        dropped_df = sm[sm["banned"] == True][["coin", "tf", "ban_time", "ban_reason", "trades", "net_pnl"]]  # noqa: E712
        extra["Dropped coins"] = dropped_df if len(dropped_df) else pd.DataFrame({"info": ["no coin was dropped"]})
        print("\nROTATION TOTAL (per timeframe)\n" + rot_tot.to_string(index=False))
    if len(sm) and len(coins) > 1:
        cx = smd.pivot_table(index="coin", columns="tf", values="net_pnl", aggfunc="sum")
        cx = cx[[c for c in TFS if c in cx.columns]]
        cx["total"] = cx.sum(axis=1)
        extra["Coin x TF"] = cx.reset_index().sort_values("total", ascending=False)
    settings = dict(run_id=run_id, run_time_utc=ts.strftime("%Y-%m-%d %H:%M"), label=label,
                    coins_tested=", ".join(c.split("/")[0] for c in coins), price_type=a.price,
                    mode=a.mode, days=a.days, pivot_length=a.length, leverage=LEV, margin_rs=MARGIN,
                    position_size_rs=MARGIN * LEV, fee_pct_per_side=FEE_PCT, stop_loss=sl_text(),
                    take_profit_pct=TP, account_balance_rs=(f"{a.balance:g} per coin" if hold else "n/a (isolated)"),
                    min_open=a.min_open if hold else "n/a", max_open=a.max_open if hold else "n/a",
                    timeframes=a.tfs, no_chase=("rotation: failed coin dropped" if rot else a.no_chase),
                    rotation_slots=rot if rot else "off",
                    scan=("off" if not scan else
                          (f">= {scan['pct']:g}% PUMP or DUMP within {scan['window']}h" if scan["window"] else
                           f">= {scan['pct']:g}% pump AND dump (whole period)")
                          + f" on {scan['exchange']}; {n_scanned} scanned, {len(scan_df)} found, {len(coins)} tested"))
    write_excel(base + ".xlsx", smd, at, pv, settings, extra)
    print(f"\nFiles saved in {a.out}/ with prefix {run_id}")
    gh_summary(sm if len(sm) else None, a, label, run_id, scan, scan_df, n_scanned, phase_tot, phase_type_gh, rot_tot)

# END_OF_SCRIPT (agar ye line file ke aakhir me nahi hai to copy adhoori hai)
