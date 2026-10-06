#!/usr/bin/env python3
"""
Download Shark Exchange historical OHLCV candles for the backtest.

Shark public endpoint:
POST https://api.sharkexchange.in/v1/market/klines?priceType=LAST_PRICE

No API key is required for market klines.

The downloader paginates with limit=1000 so the backtest never receives
a one-candle sample file by mistake.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.sharkexchange.in/v1/market/klines"
MAX_LIMIT = 1000
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}


def extract_rows(payload):
    # Shark documentation shows the endpoint returning a JSON list.
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "result", "klines", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise RuntimeError(f"Unexpected kline response shape: {type(payload).__name__}")


def normalize(rows):
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue

        # Documented Shark fields.
        if "startTime" in r:
            ts = int(r["startTime"])
            out.append({
                "timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r.get("volume", 0)),
                "endTime": int(r.get("endTime", ts)),
            })
    return out


def download(pair: str, interval: str, days: int, price_type: str):
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    end_ms = int(time.time() * 1000)
    start_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )

    step = INTERVAL_MS[interval]
    cursor = start_ms
    all_rows = []
    page = 0

    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "User-Agent": "luxalgo-pivot-backtest/1.0",
    })

    while cursor < end_ms:
        page += 1
        body = {
            "pair": pair.upper(),
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }

        r = session.post(
            f"{URL}?priceType={price_type}",
            json=body,
            timeout=30,
        )
        r.raise_for_status()

        rows = extract_rows(r.json())
        if not rows:
            print(f"Page {page}: no more candles.")
            break

        normalized = normalize(rows)
        if not normalized:
            raise RuntimeError(
                f"Page {page}: API returned rows but none had documented "
                f"startTime/open/high/low/close fields."
            )

        all_rows.extend(normalized)

        last_ts = max(x["timestamp"] for x in normalized)
        next_cursor = int(last_ts.timestamp() * 1000) + step

        print(
            f"Page {page}: {len(normalized)} candles | "
            f"through {last_ts.isoformat()}"
        )

        if next_cursor <= cursor:
            raise RuntimeError("API pagination did not advance.")
        cursor = next_cursor

        # Be polite to the public endpoint.
        time.sleep(0.15)

        # If API returned fewer than the page limit, this is normally the end.
        if len(normalized) < MAX_LIMIT:
            break

    if not all_rows:
        raise RuntimeError(
            f"No candles returned for {pair} {interval}. "
            "Check the symbol and Shark Exchange market availability."
        )

    df = pd.DataFrame(all_rows)
    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .reset_index(drop=True)
    )

    # Remove any still-open current candle.
    now_ms = int(time.time() * 1000)
    df = df[df["endTime"] < now_ms].copy()

    # Need enough candles for a 50/50 pivot and meaningful backtest.
    if len(df) < 200:
        raise RuntimeError(
            f"Downloaded only {len(df)} CLOSED candles. "
            "Need at least 200. Increase --days or check the pair."
        )

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--price-type", choices=["LAST_PRICE", "MARK_PRICE"],
                    default="LAST_PRICE")
    ap.add_argument("--output", default="data/BTCUSDT_5m.csv")
    args = ap.parse_args()

    df = download(args.pair, args.interval, args.days, args.price_type)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n========== DOWNLOAD COMPLETE ==========")
    print(f"Pair:       {args.pair}")
    print(f"Interval:   {args.interval}")
    print(f"Price type: {args.price_type}")
    print(f"Candles:    {len(df):,}")
    print(f"From:       {df['timestamp'].iloc[0]}")
    print(f"To:         {df['timestamp'].iloc[-1]}")
    print(f"Saved:      {out}")


if __name__ == "__main__":
    main()
