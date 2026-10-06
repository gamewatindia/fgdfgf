#!/usr/bin/env python3
"""
LuxAlgo-style Pivot + Missed Reversal Backtest
------------------------------------------------
Backtest only. NO live orders are placed.

Strategy:
- Timeframe default: 5m
- Pivot length default: 50
- Margin per trade: INR 1,000
- Leverage default: 10x
- LONG:
    * Price tests a support level (regular pivot low or missed low)
    * A bullish confirmation candle closes
    * Entry when a later candle breaks confirmation candle HIGH
    * SL = confirmation candle LOW
- SHORT:
    * Price tests a resistance level (regular pivot high or missed high)
    * A bearish confirmation candle closes
    * Entry when a later candle breaks confirmation candle LOW
    * SL = confirmation candle HIGH
- Default target = next opposite structural level when it gives >= MIN_TARGET_RR.
  If no valid structural target exists, fallback to FIXED_RR.
- Fixed-RR comparison is also printed for 1.0R, 1.5R, 2.0R and 3.0R.
- Pivots are confirmed only after `length` future bars. This avoids look-ahead bias.

CSV format:
timestamp,open,high,low,close,volume
2026-01-01T00:00:00Z,100,101,99,100.5,1234
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


# =========================
# CONFIG
# =========================
PIVOT_LENGTH = 50
MARGIN_INR = 1000.0
LEVERAGE = 10.0

TOUCH_TOLERANCE_PCT = 0.0015       # 0.15%
MIN_TARGET_RR = 1.50
FALLBACK_RR = 2.00

TAKER_FEE = 0.00040                # 0.040% per side
GST_ON_FEE = 0.18                  # 18% GST on trading fee
SLIPPAGE = 0.00020                 # 0.02% per side

MAX_BARS_IN_TRADE = 2000


@dataclass
class Level:
    price: float
    kind: str                 # support / resistance
    source: str               # regular / missed
    bar: int
    active: bool = True


@dataclass
class Trade:
    symbol: str
    side: str
    signal_bar: int
    entry_bar: int
    exit_bar: int
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    stop: float
    target: float
    exit: float
    qty: float
    margin: float
    leverage: float
    pnl_gross: float
    fees: float
    pnl_net: float
    rr_planned: float
    exit_reason: str
    level_source: str
    level_price: float


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    if len(df) < PIVOT_LENGTH * 2 + 100:
        raise ValueError(
            f"Not enough candles. Need at least ~{PIVOT_LENGTH * 2 + 100}, "
            f"got {len(df)}."
        )

    return df


def is_pivot_high(df: pd.DataFrame, pivot_i: int, length: int) -> bool:
    p = float(df.at[pivot_i, "high"])
    left = df["high"].iloc[pivot_i - length:pivot_i]
    right = df["high"].iloc[pivot_i + 1:pivot_i + length + 1]
    return p >= float(left.max()) and p >= float(right.max())


def is_pivot_low(df: pd.DataFrame, pivot_i: int, length: int) -> bool:
    p = float(df.at[pivot_i, "low"])
    left = df["low"].iloc[pivot_i - length:pivot_i]
    right = df["low"].iloc[pivot_i + 1:pivot_i + length + 1]
    return p <= float(left.min()) and p <= float(right.min())


def fee_for_notional(notional: float) -> float:
    trading_fee = notional * TAKER_FEE
    return trading_fee * (1.0 + GST_ON_FEE)


def apply_entry_slippage(price: float, side: str) -> float:
    return price * (1 + SLIPPAGE) if side == "LONG" else price * (1 - SLIPPAGE)


def apply_exit_slippage(price: float, side: str) -> float:
    # Exit from LONG = sell; exit from SHORT = buy.
    return price * (1 - SLIPPAGE) if side == "LONG" else price * (1 + SLIPPAGE)


def structural_target(
    side: str,
    entry: float,
    stop: float,
    levels: list[Level],
) -> tuple[Optional[float], Optional[Level], float]:
    risk = abs(entry - stop)
    if risk <= 0:
        return None, None, 0.0

    candidates = []
    if side == "LONG":
        for lv in levels:
            if lv.active and lv.kind == "resistance" and lv.price > entry:
                rr = (lv.price - entry) / risk
                if rr >= MIN_TARGET_RR:
                    candidates.append((lv.price, lv, rr))
        if not candidates:
            return None, None, 0.0
        return min(candidates, key=lambda x: x[0])

    for lv in levels:
        if lv.active and lv.kind == "support" and lv.price < entry:
            rr = (entry - lv.price) / risk
            if rr >= MIN_TARGET_RR:
                candidates.append((lv.price, lv, rr))
    if not candidates:
        return None, None, 0.0
    return max(candidates, key=lambda x: x[0])


def run_backtest(df: pd.DataFrame, symbol: str, fixed_rr: Optional[float] = None):
    levels: list[Level] = []
    trades: list[Trade] = []

    # Pending setup waits for breakout of the confirmation candle.
    pending = None
    open_trade = None

    # These are the "extreme between confirmed pivots" trackers used to
    # approximate LuxAlgo's missed-reversal concept without future leakage.
    running_high = -math.inf
    running_low = math.inf
    running_high_bar = None
    running_low_bar = None

    last_pivot_type = None
    last_pivot_price = None

    start = PIVOT_LENGTH * 2
    n = len(df)

    for i in range(start, n):
        row = df.iloc[i]

        # ------------------------------------------------------------
        # 1) Confirm pivot that occurred `length` bars ago.
        #    At current i, pivot_i = i - length is now knowable.
        # ------------------------------------------------------------
        pivot_i = i - PIVOT_LENGTH

        if is_pivot_high(df, pivot_i, PIVOT_LENGTH):
            p = float(df.at[pivot_i, "high"])

            # Missed low = lowest low between previous confirmed pivot
            # and this new confirmed high.
            if last_pivot_type == "high" and running_low < math.inf:
                levels.append(Level(
                    price=running_low,
                    kind="support",
                    source="missed",
                    bar=int(running_low_bar),
                ))

            levels.append(Level(
                price=p,
                kind="resistance",
                source="regular",
                bar=pivot_i,
            ))

            last_pivot_type = "high"
            last_pivot_price = p

            running_high = p
            running_high_bar = pivot_i
            running_low = float(df.at[pivot_i, "low"])
            running_low_bar = pivot_i

        if is_pivot_low(df, pivot_i, PIVOT_LENGTH):
            p = float(df.at[pivot_i, "low"])

            # Missed high = highest high between previous confirmed pivot
            # and this new confirmed low.
            if last_pivot_type == "low" and running_high > -math.inf:
                levels.append(Level(
                    price=running_high,
                    kind="resistance",
                    source="missed",
                    bar=int(running_high_bar),
                ))

            levels.append(Level(
                price=p,
                kind="support",
                source="regular",
                bar=pivot_i,
            ))

            last_pivot_type = "low"
            last_pivot_price = p

            running_low = p
            running_low_bar = pivot_i
            running_high = float(df.at[pivot_i, "high"])
            running_high_bar = pivot_i

        # Update extremes using only candles that are already known.
        if float(row["high"]) > running_high:
            running_high = float(row["high"])
            running_high_bar = i
        if float(row["low"]) < running_low:
            running_low = float(row["low"])
            running_low_bar = i

        # ------------------------------------------------------------
        # 2) Manage an existing trade.
        # Conservative same-candle rule:
        # if SL and TP are both touched, assume SL happened first.
        # ------------------------------------------------------------
        if open_trade is not None:
            ot = open_trade
            bars_held = i - ot["entry_bar"]

            hi = float(row["high"])
            lo = float(row["low"])

            hit_sl = lo <= ot["stop"] if ot["side"] == "LONG" else hi >= ot["stop"]
            hit_tp = hi >= ot["target"] if ot["side"] == "LONG" else lo <= ot["target"]

            reason = None
            raw_exit = None

            if hit_sl:
                reason = "SL"
                raw_exit = ot["stop"]
            elif hit_tp:
                reason = "TARGET"
                raw_exit = ot["target"]
            elif bars_held >= MAX_BARS_IN_TRADE:
                reason = "TIME"
                raw_exit = float(row["close"])

            if reason:
                exit_price = apply_exit_slippage(raw_exit, ot["side"])
                if ot["side"] == "LONG":
                    gross = (exit_price - ot["entry"]) * ot["qty"]
                else:
                    gross = (ot["entry"] - exit_price) * ot["qty"]

                exit_notional = exit_price * ot["qty"]
                fees = ot["entry_fee"] + fee_for_notional(exit_notional)

                trades.append(Trade(
                    symbol=symbol,
                    side=ot["side"],
                    signal_bar=ot["signal_bar"],
                    entry_bar=ot["entry_bar"],
                    exit_bar=i,
                    signal_time=df.at[ot["signal_bar"], "timestamp"].isoformat(),
                    entry_time=df.at[ot["entry_bar"], "timestamp"].isoformat(),
                    exit_time=df.at[i, "timestamp"].isoformat(),
                    entry=ot["entry"],
                    stop=ot["stop"],
                    target=ot["target"],
                    exit=exit_price,
                    qty=ot["qty"],
                    margin=MARGIN_INR,
                    leverage=LEVERAGE,
                    pnl_gross=gross,
                    fees=fees,
                    pnl_net=gross - fees,
                    rr_planned=ot["rr_planned"],
                    exit_reason=reason,
                    level_source=ot["level_source"],
                    level_price=ot["level_price"],
                ))

                open_trade = None
                pending = None
                continue

        # ------------------------------------------------------------
        # 3) If there is a pending confirmation, wait for breakout.
        # ------------------------------------------------------------
        if open_trade is None and pending is not None:
            if i > pending["signal_bar"]:
                if pending["side"] == "LONG" and float(row["high"]) > pending["trigger"]:
                    entry_raw = pending["trigger"]
                    entry = apply_entry_slippage(entry_raw, "LONG")
                    stop = pending["stop"]

                    risk = entry - stop
                    if risk > 0:
                        target = pending["target"]
                        rr = (target - entry) / risk
                        if target > entry and rr >= 0.5:
                            qty = (MARGIN_INR * LEVERAGE) / entry
                            entry_fee = fee_for_notional(entry * qty)

                            open_trade = {
                                "side": "LONG",
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "qty": qty,
                                "entry_fee": entry_fee,
                                "entry_bar": i,
                                "signal_bar": pending["signal_bar"],
                                "rr_planned": rr,
                                "level_source": pending["level_source"],
                                "level_price": pending["level_price"],
                            }
                            pending = None

                elif pending["side"] == "SHORT" and float(row["low"]) < pending["trigger"]:
                    entry_raw = pending["trigger"]
                    entry = apply_entry_slippage(entry_raw, "SHORT")
                    stop = pending["stop"]

                    risk = stop - entry
                    if risk > 0:
                        target = pending["target"]
                        rr = (entry - target) / risk
                        if target < entry and rr >= 0.5:
                            qty = (MARGIN_INR * LEVERAGE) / entry
                            entry_fee = fee_for_notional(entry * qty)

                            open_trade = {
                                "side": "SHORT",
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "qty": qty,
                                "entry_fee": entry_fee,
                                "entry_bar": i,
                                "signal_bar": pending["signal_bar"],
                                "rr_planned": rr,
                                "level_source": pending["level_source"],
                                "level_price": pending["level_price"],
                            }
                            pending = None

        # ------------------------------------------------------------
        # 4) Create a new confirmation setup.
        # We only use levels known BEFORE this candle.
        # ------------------------------------------------------------
        if open_trade is None and pending is None:
            bullish = float(row["close"]) > float(row["open"])
            bearish = float(row["close"]) < float(row["open"])
            hi = float(row["high"])
            lo = float(row["low"])
            close = float(row["close"])

            support_candidates = [
                lv for lv in levels
                if lv.active and lv.kind == "support"
                and abs(lo - lv.price) / lv.price <= TOUCH_TOLERANCE_PCT
                and close > lv.price
            ]

            resistance_candidates = [
                lv for lv in levels
                if lv.active and lv.kind == "resistance"
                and abs(hi - lv.price) / lv.price <= TOUCH_TOLERANCE_PCT
                and close < lv.price
            ]

            # Prefer the nearest level.
            if bullish and support_candidates:
                lv = max(support_candidates, key=lambda x: x.price)
                stop = lo

                if stop < close:
                    if fixed_rr is not None:
                        target = close + (close - stop) * fixed_rr
                    else:
                        st, _, _ = structural_target("LONG", close, stop, levels)
                        target = st if st is not None else close + (close - stop) * FALLBACK_RR

                    pending = {
                        "side": "LONG",
                        "signal_bar": i,
                        "trigger": hi,
                        "stop": stop,
                        "target": target,
                        "level_source": lv.source,
                        "level_price": lv.price,
                    }

            elif bearish and resistance_candidates:
                lv = min(resistance_candidates, key=lambda x: x.price)
                stop = hi

                if stop > close:
                    if fixed_rr is not None:
                        target = close - (stop - close) * fixed_rr
                    else:
                        st, _, _ = structural_target("SHORT", close, stop, levels)
                        target = st if st is not None else close - (stop - close) * FALLBACK_RR

                    pending = {
                        "side": "SHORT",
                        "signal_bar": i,
                        "trigger": lo,
                        "stop": stop,
                        "target": target,
                        "level_source": lv.source,
                        "level_price": lv.price,
                    }

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    return trades_df


def summary(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {
            "mode": label,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0,
            "net_pnl_inr": 0,
            "gross_pnl_inr": 0,
            "fees_inr": 0,
            "profit_factor": 0,
            "max_drawdown_inr": 0,
        }

    wins = trades[trades["pnl_net"] > 0]
    losses = trades[trades["pnl_net"] <= 0]
    gross_profit = float(wins["pnl_net"].sum())
    gross_loss = abs(float(losses["pnl_net"].sum()))
    pf = gross_profit / gross_loss if gross_loss else math.inf

    equity = trades["pnl_net"].cumsum()
    peak = equity.cummax()
    dd = equity - peak

    return {
        "mode": label,
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2),
        "net_pnl_inr": round(float(trades["pnl_net"].sum()), 2),
        "gross_pnl_inr": round(float(trades["pnl_gross"].sum()), 2),
        "fees_inr": round(float(trades["fees"].sum()), 2),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else "INF",
        "max_drawdown_inr": round(abs(float(dd.min())), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="OHLCV CSV")
    ap.add_argument("--symbol", default="UNKNOWN")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.data)

    print(f"\nLoaded {len(df):,} candles")
    print(f"Period: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"Pivot length: {PIVOT_LENGTH}")
    print(f"Margin/trade: INR {MARGIN_INR:,.0f}")
    print(f"Leverage: {LEVERAGE:g}x")
    print(f"Position notional: INR {MARGIN_INR * LEVERAGE:,.0f}")

    all_summaries = []

    # Structural target mode
    trades = run_backtest(df, args.symbol, fixed_rr=None)
    trades.to_csv(out / "trades_structural.csv", index=False)
    all_summaries.append(summary(trades, "STRUCTURAL"))

    # Fixed RR comparison
    for rr in [1.0, 1.5, 2.0, 3.0]:
        t = run_backtest(df, args.symbol, fixed_rr=rr)
        t.to_csv(out / f"trades_rr_{str(rr).replace('.', '_')}.csv", index=False)
        all_summaries.append(summary(t, f"{rr}:1"))

    result = pd.DataFrame(all_summaries)
    result.to_csv(out / "summary.csv", index=False)

    print("\n========== BACKTEST SUMMARY ==========")
    print(result.to_string(index=False))
    print("\nTrade files written to:", out.resolve())


if __name__ == "__main__":
    main()
