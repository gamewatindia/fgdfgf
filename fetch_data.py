#!/usr/bin/env python3
"""
Download 5m OHLCV history from MEXC or Gate.io public APIs (no API key needed)
and save as data/SYMBOL.csv for brahmastra_backtest.py.

Usage (run on YOUR computer, needs: pip install requests pandas):
  python fetch_data.py --exchange mexc --symbols BTC_USDT,ETH_USDT,SOL_USDT --days 365
  python fetch_data.py --exchange gate --symbols BTC_USDT,ETH_USDT --days 365

Symbols are written like BASE_QUOTE. Downloads 5m candles (the backtester builds
15m/1h/4h/1d/1w/1M from them). 365 days of 5m = ~105k candles per coin.
NOTE: written from the exchanges' public API docs; not run against the live
APIs from the build environment (blocked there) - if an error shows up, send it.
"""
import argparse
import os
import time

import pandas as pd
import requests

S = requests.Session()
S.headers["User-Agent"] = "brahmastra-backtest/1.0"


def mexc(sym, start_ms, end_ms):
    url = "https://api.mexc.com/api/v3/klines"
    pair = sym.replace("_", "")
    rows, t = [], start_ms
    while t < end_ms:
        r = S.get(url, params=dict(symbol=pair, interval="5m", startTime=t, limit=500), timeout=20)
        r.raise_for_status()
        k = r.json()
        if not k:
            break
        rows += [(x[0], x[1], x[2], x[3], x[4], x[5]) for x in k]
        t = k[-1][0] + 5 * 60 * 1000
        time.sleep(0.12)
    return rows


def gate(sym, start_ms, end_ms):
    url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    rows, t = [], start_ms // 1000
    step = 5 * 60 * 900  # 900 candles per request (max 1000)
    while t < end_ms // 1000:
        to = min(t + step, end_ms // 1000)
        r = S.get(url, params={"currency_pair": sym, "interval": "5m", "from": t, "to": to}, timeout=20)
        r.raise_for_status()
        # Gate row: [ts, quote_vol, close, high, low, open, base_vol, closed]
        rows += [(int(x[0]) * 1000, x[5], x[3], x[4], x[2], x[6]) for x in r.json()]
        t = to + 1
        time.sleep(0.12)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML settings file (see config.yml)")
    ap.add_argument("--exchange", choices=["mexc", "gate"])
    ap.add_argument("--symbols", help="comma list, e.g. BTC_USDT,ETH_USDT")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    if a.config:
        import yaml
        with open(a.config) as fh:
            d = (yaml.safe_load(fh) or {}).get("data", {})
        a.exchange = a.exchange or d.get("exchange")
        a.symbols = a.symbols or ",".join(d.get("symbols", []))
        a.days = d.get("days", a.days) if a.days == 365 else a.days
        a.out = d.get("folder", a.out) if a.out == "data" else a.out
    if not a.exchange or not a.symbols:
        ap.error("need --exchange and --symbols (or --config)")
    os.makedirs(a.out, exist_ok=True)
    end = int(time.time() * 1000)
    start = end - a.days * 86400 * 1000
    fn = mexc if a.exchange == "mexc" else gate
    for sym in [s.strip().upper() for s in a.symbols.split(",")]:
        try:
            rows = fn(sym, start, end)
        except Exception as e:
            print(f"{sym}: FAILED -> {e}")
            continue
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        df.to_csv(os.path.join(a.out, f"{sym.replace('_', '')}_{a.exchange}.csv"), index=False)
        print(f"{sym}: {len(df)} candles saved")


if __name__ == "__main__":
    main()
