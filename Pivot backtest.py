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

PUMP & DUMP SCAN (--scan "40,gateio,300,15")
  Scans the exchange's USDT coins for the same timeline (--days). A coin qualifies if it PUMPED at least
  X% (low -> later high) AND then DUMPED at least X% from that peak. Only the qualifying coins are then
  backtested (all timeframes, same settings). Format: pct[,exchange[,max_coins_to_scan[,max_coins_to_test]]]
  --pair ALL = whole exchange, --pair A,B,C = only scan these, one pair = just check that coin.

Every output file name + every row carries run_id / coin / pair / exchange so uploads are self-explaining.
Outputs (folder results/): <run_id>_summary.csv, _trades.csv, _pivots.csv, _scan.csv, _equity.png, <run_id>.xlsx
Run: python pivot_backtest.py --pair BTCUSDT --days 30 --mode hold_long --tp 5 --balance 10000 --max_open 3
Pair can carry a preferred exchange:  --pair BRUSDT@mexc
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
HOLD_ONLY_COLS = ["account_liq", "min_equity", "final_equity", "skipped", "skipped_cap", "peak_open"]


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
    order = ([first] if first else []) + [e for e in EXCHANGES[price] if e != first]
    for name in order:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
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


def parse_scan(s):
    """'0' -> None (off).  '40' | '40,gateio' | '40,gateio,300,15' -> settings dict."""
    parts = [p.strip() for p in str(s).split(",")]
    if not parts or parts[0] in ("", "0", "off", "OFF"):
        return None
    pct = float(parts[0])
    if pct <= 0:
        return None
    ex = parts[1].lower() if len(parts) > 1 and parts[1] else "gateio"
    mx_scan = int(parts[2]) if len(parts) > 2 and parts[2] else 300
    mx_test = int(parts[3]) if len(parts) > 3 and parts[3] else 15
    return dict(pct=pct, exchange=ex, max_scan=mx_scan, max_test=mx_test)


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


def detect_pump_dump(df, pct):
    """Best 'pump then dump': for every candle j as the PEAK -> pump = peak / lowest low before it - 1,
    dump = 1 - lowest low after it / peak. A coin qualifies if min(pump, dump) >= pct."""
    if len(df) < 20:
        return None
    h, l, t = df["high"].values, df["low"].values, df["time"]
    pump = h / np.minimum.accumulate(l) - 1
    suf = np.full(len(l), np.inf)
    suf[:-1] = np.minimum.accumulate(l[::-1])[::-1][1:]          # lowest low strictly after j
    dump = np.where(np.isfinite(suf), 1 - suf / h, 0.0)
    j = int(np.argmax(np.minimum(pump, dump)))
    if min(pump[j], dump[j]) < pct / 100:
        return None
    i = int(np.argmin(l[: j + 1]))
    k = j + 1 + int(np.argmin(l[j + 1:]))
    return dict(pump_pct=round(pump[j] * 100, 1), dump_pct=round(dump[j] * 100, 1),
                low_time=str(t[i]), low_price=l[i], peak_time=str(t[j]), peak_price=h[j],
                dump_low_time=str(t[k]), dump_low_price=l[k])


def scan_pump_dump(pairs, scan, days):
    """Returns (found_df, n_scanned). pairs=None -> every active USDT spot coin on the exchange."""
    import ccxt
    ex = getattr(ccxt, scan["exchange"])({"enableRateLimit": True})
    ex.load_markets()
    tf = "1h" if days <= 60 else "4h"
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
    print(f"scan: checking {len(syms)} coins on {scan['exchange']} ({tf} candles, last {days} days) "
          f"for >= {scan['pct']:g}% pump AND >= {scan['pct']:g}% dump")
    found = []
    for n, sym in enumerate(syms, 1):
        try:
            r = detect_pump_dump(fetch_simple(ex, sym, tf, days), scan["pct"])
        except Exception:
            continue
        if r:
            found.append(dict(pair=sym, coin=sym.split("/")[0], **r, scan_tf=tf, scan_exchange=scan["exchange"]))
        if n % 50 == 0:
            print(f"scan: {n}/{len(syms)} checked, {len(found)} found")
    df = pd.DataFrame(found)
    if len(df):
        df["move_pct"] = df[["pump_pct", "dump_pct"]].min(axis=1)
        df = df.sort_values("move_pct", ascending=False).reset_index(drop=True)
    return df, len(syms)


def get_signals(df, length, days, at_pivot):
    """{bar_index: (direction, price)}. direction +1 = pivot low (long), -1 = pivot high (short).
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
        sig[k] = (-1 if ph else 1, px)
    return sig


def excursion(d, epx, hi, lo):
    """(adverse, favourable) move of this bar vs entry, fraction of price. adverse > 0 = against you."""
    if d == 1:
        return (epx - lo) / epx, (hi - epx) / epx
    return (hi - epx) / epx, (epx - lo) / epx


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


def reverse_backtest(df, length, days, tf, at_pivot):
    high, low, t = df["high"].values, df["low"].values, df["time"]
    sig = get_signals(df, length, days, at_pivot)
    notional = MARGIN * LEV
    liq = max(1 / LEV - MMR, 0.001)                  # adverse move that wipes the margin
    sd = SL / 100 if 0 < SL / 100 < liq else None    # stop only useful if before liquidation
    tp = TP / 100 if TP > 0 else None
    trades, pos, epx, et, mae, mfe = [], 0, None, None, 0.0, 0.0

    def close_trade(px, tm, status):
        g = notional * pos * (px / epx - 1)
        fee = notional * FEE_PCT / 100 * 2
        net = max(g - fee, -MARGIN)                  # loss can never exceed the margin
        if status == "LIQUIDATED":
            net = -MARGIN                            # liquidation = full margin lost
        trades.append(dict(tf=tf, side="LONG" if pos == 1 else "SHORT", entry_time=et,
                           entry_price=epx, exit_time=tm, exit_price=px,
                           mfe_pct=round(mfe * 100, 2), mae_pct=round(mae * 100, 2),
                           hold_h=round((tm - et).total_seconds() / 3600, 1),
                           gross_pnl=round(max(g, -MARGIN), 2), fees=round(fee, 2),
                           net_pnl=round(net, 2), status=status))

    for j in range(len(df)):
        if pos:
            adv, fav = excursion(pos, epx, high[j], low[j])
            mae, mfe = max(mae, adv), max(mfe, fav)
            if sd and adv >= sd:
                close_trade(epx * (1 - pos * sd), t.iloc[j], "STOP LOSS"); pos = 0
            elif adv >= liq:
                close_trade(epx * (1 - pos * liq), t.iloc[j], "LIQUIDATED"); pos = 0
            elif tp and fav >= tp:
                close_trade(epx * (1 + pos * tp), t.iloc[j], "TARGET"); pos = 0
        if j in sig:
            d, px = sig[j]
            if d == pos:
                continue
            if pos:
                close_trade(px, t.iloc[j], "closed")
            pos, epx, et, mae, mfe = d, px, t.iloc[j], 0.0, 0.0
    if pos:
        close_trade(df["close"].iloc[-1], t.iloc[-1], "open (MTM)")
    return pd.DataFrame(trades), len(sig), {}


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
            d, px = sig[j]
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
             final_equity="", skipped="", skipped_cap="", peak_open="", note=err)
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
    ("status (trades)", "TARGET, closed (opposite pivot), STOP LOSS, LIQUIDATED, ACCOUNT LIQUIDATED, "
                        "DEFERRED EXIT (min_open), open (MTM)"),
    ("mfe_pct / mae_pct / hold_h", "Best move for / worst move against the trade (% of price); hours held"),
    ("min_open / max_open", "Hold modes: minimum trades kept running / maximum trades at the same time (0 = off)"),
    ("Scan sheet: pump_pct / dump_pct", "Rise from the lowest low to the peak / fall from that peak to the lowest "
                                        "low after it. Coin qualifies if BOTH are >= the scan %"),
    ("Scan sheet: low/peak/dump_low", "Time and price of the pump start, the peak, and the dump bottom"),
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


def gh_summary(sm, args, label, run_id, scan, scan_df, n_scanned):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    hold = args.mode.startswith("hold")
    with open(path, "a") as f:
        f.write(f"## {label} | {args.mode} | {args.days}d | {LEV:g}x | SL {SL:g}% | TP {TP:g}%"
                f"{f' | open {args.min_open}-{args.max_open}' if hold else ''} | pivot length {args.length}\n\n")
        f.write(f"`run_id: {run_id}`\n\n")
        if scan:
            n = 0 if scan_df is None else len(scan_df)
            f.write(f"**Pump & dump scan** ({scan['exchange']}, >= {scan['pct']:g}% pump AND dump): "
                    f"{n_scanned} coins checked, **{n} found**, {min(n, scan['max_test'])} tested\n\n")
            if n:
                f.write("| Coin | Pump % | Dump % | Pump start | Peak | Dump bottom |\n|---|---|---|---|---|---|\n")
                for r in scan_df.head(scan["max_test"]).itertuples():
                    f.write(f"| {r.coin} | {r.pump_pct} | {r.dump_pct} | {r.low_time[:16]} | "
                            f"{r.peak_time[:16]} | {r.dump_low_time[:16]} |\n")
                f.write("\n")
        if args.mode == "reverse_pivot_price":
            f.write("> NOT TRADABLE: entry at the real pivot price needs future data. Comparison only.\n\n")
        if sm is None or not len(sm):
            f.write("No coins to test.\n")
            return
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


def run_coin(pair, a, first, run_id, ts, hold):
    """Backtest ONE coin on all timeframes -> (summary rows, trades df|None, pivots df|None)."""
    coin = pair.split("/")[0]
    rows, trs, pvs, ex_by_tf = [], [], [], {}
    for tf in a.tfs.split(","):
        info = {}
        try:
            df, info = fetch(pair, tf, a.days, 2 * a.length + 5, a.price, first)
            ex_by_tf[tf] = info["exchange"]
            pvs.append(list_pivots(df, a.length, a.days, tf))
            if hold:
                side = {"hold_long": 1, "hold_short": -1, "hold_both": 0}[a.mode]
                tr, npv, extra = hold_backtest(df, a.length, a.days, tf, side, a.balance, a.min_open, a.max_open)
            else:
                tr, npv, extra = reverse_backtest(df, a.length, a.days, tf, a.mode == "reverse_pivot_price")
            row = summarize(tf, tr, npv, extra)
            if not tr.empty:
                trs.append(tr)
        except Exception as e:
            print(f"[{pair} {tf}] ERROR: {e}")
            row = summarize(tf, pd.DataFrame(), 0, None, "ERROR: " + str(e)[:300])
        row.update(run_id=run_id, coin=coin, pair=pair, mode=a.mode, exchange=info.get("exchange", ""),
                   price_type=a.price, days=a.days, leverage=LEV, margin_rs=MARGIN, sl_pct=SL, tp_pct=TP,
                   balance_rs=a.balance if hold else "", min_open=a.min_open if hold else "",
                   max_open=a.max_open if hold else "", pivot_length=a.length, fee_pct=FEE_PCT,
                   candles=info.get("candles", ""), days_covered=info.get("days_covered", ""),
                   data_from=info.get("data_from", ""), data_to=info.get("data_to", ""),
                   run_time_utc=ts.strftime("%Y-%m-%d %H:%M"))
        rows.append(row)

    def tag(d):                                   # run/coin/pair/exchange in front of every row
        d = d.copy()
        d.insert(0, "run_id", run_id); d.insert(1, "coin", coin); d.insert(2, "pair", pair)
        d.insert(3, "exchange", d["tf"].map(ex_by_tf)); d.insert(4, "mode", a.mode)
        return d

    trades = tag(pd.concat(trs, ignore_index=True)) if trs else None
    pivots = tag(pd.concat(pvs, ignore_index=True)) if pvs and any(len(p) for p in pvs) else None
    return rows, trades, pivots


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BTCUSDT",
                    help="BTCUSDT | BRUSDT@mexc (preferred exchange) | with --scan: ALL or A,B,C")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--price", default="LAST_PRICE", choices=list(EXCHANGES))
    ap.add_argument("--length", type=int, default=50)
    ap.add_argument("--tfs", default=",".join(TFS))
    ap.add_argument("--first", default="", help="preferred exchange tried first (ccxt id)")
    ap.add_argument("--lev", type=float, default=10)
    ap.add_argument("--margin", type=float, default=1000)
    ap.add_argument("--sl", type=float, default=0, help="stop-loss %% of price, 0 = off")
    ap.add_argument("--tp", type=float, default=0, help="take-profit %% of price, 0 = off")
    ap.add_argument("--mode", default="reverse", choices=MODES)
    ap.add_argument("--balance", type=float, default=10000, help="cross-margin account balance (hold modes)")
    ap.add_argument("--min_open", type=int, default=0, help="hold modes: keep at least N trades running (0 = off)")
    ap.add_argument("--max_open", type=int, default=0, help="hold modes: at most N trades at once (0 = no cap)")
    ap.add_argument("--scan", default="0", help='pump&dump scan: "0" = off, or "40,gateio,300,15" = '
                    "pct,exchange,max_scan,max_test")
    ap.add_argument("--out", default="results", help="output folder")
    a = ap.parse_args()
    LEV, MARGIN, SL, TP = a.lev, a.margin, a.sl, a.tp
    hold = a.mode.startswith("hold")
    if a.min_open < 0 or a.max_open < 0:
        ap.error("min_open / max_open cannot be negative")
    if a.max_open and a.min_open >= a.max_open:
        ap.error("min_open must be smaller than max_open (else no trade could ever close)")
    try:
        scan = parse_scan(a.scan)
    except ValueError:
        ap.error('scan must look like "0" (off) or "40" or "40,gateio,300,15"')
    if hold and TP <= 0:
        TP = 5.0
        print("hold mode needs a target -> using TP = 5% of price")
    if not hold and (a.min_open or a.max_open):
        print("note: min_open / max_open apply to hold modes only (reverse modes hold 1 trade at a time)")
    raw_pair, first = a.pair.strip(), a.first.strip().lower()
    if "@" in raw_pair:
        raw_pair, ex_pref = raw_pair.split("@", 1)
        first = ex_pref.strip().lower() or first
    ts = datetime.now(timezone.utc)

    scan_df, n_scanned = None, 0
    if scan:
        first = scan["exchange"]                       # test the coins on the exchange they were found on
        pairs = None if raw_pair.upper() in ("", "ALL", "*") else \
            [norm_pair(p.strip()) for p in raw_pair.split(",") if p.strip()]
        scan_df, n_scanned = scan_pump_dump(pairs, scan, a.days)
        coins = list(scan_df["pair"][: scan["max_test"]]) if len(scan_df) else []
        label = f"SCAN{scan['pct']:g}pct-{len(coins)}coins"
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
    if hold and (a.min_open or a.max_open):
        parts.append(f"open{a.min_open}-{a.max_open}")
    run_id = "-".join(parts) + "-" + ts.strftime("%Y%m%d_%H%M")
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, run_id)

    all_rows, all_tr, all_pv = [], [], []
    for n, pr in enumerate(coins, 1):
        if scan:
            r0 = scan_df.iloc[n - 1]
            print(f"\n===== coin {n}/{len(coins)}: {pr}  (pump {r0.pump_pct}% / dump {r0.dump_pct}%) =====")
        rows, trades, pivots = run_coin(pr, a, first, run_id, ts, hold)
        all_rows += rows
        if trades is not None:
            all_tr.append(trades)
        if pivots is not None:
            all_pv.append(pivots)

    sm = pd.DataFrame(all_rows)
    if len(sm):
        lead = ["run_id", "coin", "pair", "mode", "tf", "exchange"]
        sm = sm[lead + [c for c in sm.columns if c not in lead + ["note"]] + ["note"]]
        smd = sm if hold else sm.drop(columns=HOLD_ONLY_COLS + ["balance_rs", "min_open", "max_open"])
    else:
        smd = pd.DataFrame({"info": ["no coins qualified for the pump & dump scan - nothing tested"]})
    at = pv = None
    if all_tr:
        at = pd.concat(all_tr, ignore_index=True)
        at.insert(at.columns.get_loc("tf") + 1, "trade_no", at.groupby(["coin", "tf"]).cumcount() + 1)
    if all_pv:
        pv = pd.concat(all_pv, ignore_index=True)

    print("\n" + "=" * 78)
    print(run_id)
    print(f"{label} | {a.price} | {a.days}d | mode {a.mode} | length {a.length} | margin Rs {MARGIN:.0f} x {LEV:g}x"
          f" | SL {SL:g}% | TP {TP:g}% | fee {FEE_PCT}%/side"
          + (f" | balance Rs {a.balance:g} | open {a.min_open}-{a.max_open}" if hold else ""))
    if a.mode == "reverse_pivot_price":
        print("NOTE: entry at real pivot price is NOT tradable (look-ahead) - comparison only.")
    print("=" * 78)
    if len(sm):
        show = ["coin", "tf", "exchange", "trades", "long", "short", "win_pct", "net_pnl", "realised", "open_mtm",
                "tp_hits", "open_n", "liq", "sl_hits", "worst_mae", "fees", "profit_factor", "max_dd"]
        if hold:
            show += ["account_liq", "min_equity", "final_equity", "peak_open", "skipped", "skipped_cap"]
        print(smd[show].to_string(index=False))
    else:
        print(smd.iloc[0, 0])
    smd.to_csv(base + "_summary.csv", index=False)
    if scan_df is not None:
        scan_df.to_csv(base + "_scan.csv", index=False)
        if len(scan_df):
            print("\nSCAN RESULT\n" + scan_df.drop(columns=["scan_exchange"]).to_string(index=False))
    if pv is not None:
        pv.to_csv(base + "_pivots.csv", index=False)
    if at is not None:
        at.to_csv(base + "_trades.csv", index=False)
        if not scan:
            print("\nALL TRADES - last 150 (full list in file)\n"
                  + at.tail(150).drop(columns=["run_id", "coin", "pair", "mode"]).to_string(index=False))
        plot_all(at, base + "_equity.png", f"{label} | {a.mode} | {a.days}d | {LEV:g}x"
                 + (f" | TP {TP:g}%" if TP else "") + (f" | SL {SL:g}%" if SL else ""))
    extra = {}
    if scan_df is not None:
        extra["Scan"] = scan_df if len(scan_df) else pd.DataFrame({"info": ["no coin qualified"]})
    if len(sm) and len(coins) > 1:
        cx = smd.pivot_table(index="coin", columns="tf", values="net_pnl", aggfunc="sum")
        cx = cx[[c for c in TFS if c in cx.columns]]
        cx["total"] = cx.sum(axis=1)
        extra["Coin x TF"] = cx.reset_index().sort_values("total", ascending=False)
    settings = dict(run_id=run_id, run_time_utc=ts.strftime("%Y-%m-%d %H:%M"), label=label,
                    coins_tested=", ".join(c.split("/")[0] for c in coins), price_type=a.price,
                    mode=a.mode, days=a.days, pivot_length=a.length, leverage=LEV, margin_rs=MARGIN,
                    position_size_rs=MARGIN * LEV, fee_pct_per_side=FEE_PCT, stop_loss_pct=SL,
                    take_profit_pct=TP, account_balance_rs=(f"{a.balance:g} per coin" if hold else "n/a (isolated)"),
                    min_open=a.min_open if hold else "n/a", max_open=a.max_open if hold else "n/a",
                    timeframes=a.tfs,
                    scan=("off" if not scan else f">= {scan['pct']:g}% pump AND dump on {scan['exchange']}; "
                          f"{n_scanned} scanned, {len(scan_df)} found, {len(coins)} tested"))
    write_excel(base + ".xlsx", smd, at, pv, settings, extra)
    print(f"\nFiles saved in {a.out}/ with prefix {run_id}")
    gh_summary(sm if len(sm) else None, a, label, run_id, scan, scan_df, n_scanned)

# END_OF_SCRIPT (agar ye line file ke aakhir me nahi hai to copy adhoori hai)
