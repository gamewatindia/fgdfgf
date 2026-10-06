
import os
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

API_URL = "https://api.sharkexchange.in/v1/market/klines"
INTERVAL = "5m"
PIVOT_LENGTH = 50
LIMIT = 1000

MARGIN = 1000.0
LEVERAGE = 10.0
NOTIONAL = MARGIN * LEVERAGE

FEE_RATE = 0.00040
GST_ON_FEE = 0.18
SLIPPAGE = 0.00020
TOUCH_TOLERANCE = 0.0015

RESULTS_DIR = "results"


def utc_now():
    return datetime.now(timezone.utc)


def ms_to_datetime(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def download_history(pair, days, price_type="LAST_PRICE"):
    print("\n" + "=" * 80)
    print("SHARK EXCHANGE HISTORICAL DOWNLOAD")
    print("=" * 80)

    now = utc_now()
    requested_start = now - timedelta(days=days)
    current_end_ms = int(now.timestamp() * 1000)
    requested_start_ms = int(requested_start.timestamp() * 1000)

    all_candles = []
    page = 1

    while True:
        print(f"\nDownloading page {page} (through: {ms_to_datetime(current_end_ms)})")

        try:
            response = requests.post(
                API_URL,
                params={"priceType": price_type},
                json={
                    "pair": pair,
                    "interval": INTERVAL,
                    "endTime": current_end_ms,
                    "limit": LIMIT,
                },
                timeout=30,
            )
        except Exception as e:
            raise RuntimeError(f"Shark API connection failed: {e}")

        print("HTTP STATUS:", response.status_code)

        if response.status_code not in (200, 201):
            print("\nAPI RESPONSE:")
            print(response.text[:3000])
            raise RuntimeError(f"Shark API returned HTTP {response.status_code}")

        try:
            raw = response.json()
        except Exception:
            print("\nRAW RESPONSE:")
            print(response.text[:3000])
            raise RuntimeError("Shark API response was not valid JSON.")

        candles = None
        if isinstance(raw, list):
            candles = raw
        elif isinstance(raw, dict):
            for key in ("data", "result", "rows", "candles", "klines"):
                value = raw.get(key)
                if isinstance(value, list):
                    candles = value
                    break
                if isinstance(value, dict):
                    for subkey in ("data", "rows", "candles", "klines", "result"):
                        subvalue = value.get(subkey)
                        if isinstance(subvalue, list):
                            candles = subvalue
                            break
                    if candles is not None:
                        break

        if candles is None:
            print("\nUNEXPECTED SHARK RESPONSE:")
            print(str(raw)[:5000])
            raise RuntimeError("Historical data could not be parsed.")

        if not candles:
            print("No more candles returned.")
            break

        normalized = []
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            try:
                start_time = candle.get("startTime")
                if start_time is None:
                    continue

                if isinstance(start_time, str):
                    stripped = start_time.strip()
                    if stripped.replace(".", "", 1).isdigit():
                        start_time = int(float(stripped))
                    else:
                        dt = pd.to_datetime(stripped, utc=True, errors="coerce")
                        if pd.isna(dt):
                            continue
                        start_time = int(dt.timestamp() * 1000)
                elif isinstance(start_time, (int, float)):
                    start_time = int(start_time)
                    if start_time < 10_000_000_000:
                        start_time *= 1000
                else:
                    continue

                normalized.append({
                    "timestamp": start_time,
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle.get("volume", 0) or 0),
                })
            except Exception:
                continue

        if not normalized:
            print("\nFirst API item:")
            print(str(candles[0])[:2000])
            raise RuntimeError("API returned candles but none could be normalized.")

        print(f"Page {page}: {len(normalized)} valid candles")
        all_candles.extend(normalized)

        oldest_ms = min(c["timestamp"] for c in normalized)
        print("Oldest candle:", ms_to_datetime(oldest_ms))

        if oldest_ms <= requested_start_ms:
            print("\nRequested historical period reached.")
            break

        next_end_ms = oldest_ms - 1
        if next_end_ms >= current_end_ms:
            raise RuntimeError("Pagination stopped moving backwards.")

        current_end_ms = next_end_ms
        page += 1
        time.sleep(0.30)

        if page > 500:
            raise RuntimeError("Pagination safety limit reached.")

    if not all_candles:
        raise RuntimeError("No historical candles downloaded.")

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp")
    df = df[df["timestamp"] >= requested_start_ms].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    current_5m_start = pd.Timestamp.now(tz="UTC").floor("5min")
    df = df[df["timestamp"] < current_5m_start].copy()
    df.reset_index(drop=True, inplace=True)

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print("Candles :", len(df))
    if len(df):
        print("From    :", df["timestamp"].iloc[0])
        print("To      :", df["timestamp"].iloc[-1])

    expected = days * 24 * 12
    coverage = len(df) / expected * 100 if expected else 0
    print("Expected approx :", expected)
    print(f"Coverage        : {coverage:.2f}%")

    if len(df) < 200:
        raise RuntimeError(f"Not enough candles downloaded: {len(df)}")

    return df


def build_luxalgo_events(df, length=PIVOT_LENGTH):
    """
    Reproduces the open-source LuxAlgo Pivot Points High Low &
    Missed Reversal Levels state machine as a causal event stream.

    Critical distinction:
      - pivot/ghost LABEL location = n-length
      - event APPEARANCE time = n (the confirmation/current bar)

    We use appearance time for backtesting, so no future-confirmed
    pivot is available before it actually confirms.
    """
    n_rows = len(df)
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()

    # Confirmed pivot values become available on the confirmation bar.
    ph_at = [None] * n_rows
    pl_at = [None] * n_rows

    for n in range(n_rows):
        p = n - length
        if p < length or p + length >= n_rows:
            continue

        window_h = highs[p - length:p + length + 1]
        window_l = lows[p - length:p + length + 1]

        if highs[p] >= window_h.max():
            ph_at[n] = float(highs[p])

        if lows[p] <= window_l.min():
            pl_at[n] = float(lows[p])

    # Pine vars.
    max_v = 0.0
    min_v = 0.0
    follow_max = 0.0
    follow_min = 0.0

    max_x1 = 0
    min_x1 = 0
    follow_max_x1 = 0
    follow_min_x1 = 0

    os = 0
    events = []

    for n in range(n_rows):
        p = n - length

        # Pine: max := max(high[length], max), etc.
        if p >= 0:
            prev_max = max_v
            prev_min = min_v
            prev_follow_max = follow_max
            prev_follow_min = follow_min

            max_v = max(highs[p], max_v)
            min_v = min(lows[p], min_v)
            follow_max = max(highs[p], follow_max)
            follow_min = min(lows[p], follow_min)

            if max_v > prev_max:
                max_x1 = p
                follow_min = lows[p]

            if min_v < prev_min:
                min_x1 = p
                follow_max = highs[p]

            if follow_min < prev_follow_min:
                follow_min_x1 = p

            if follow_max > prev_follow_max:
                follow_max_x1 = p

        ph = ph_at[n]
        pl = pl_at[n]

        # Important: os[1] means previous state.
        prev_os = os

        if ph is not None:
            # Missed reversal branch from the original script.
            if prev_os == 1:
                events.append({
                    "appearance_index": n,
                    "location_index": min_x1,
                    "type": "missed_low",
                    "price": float(min_v),
                })

            elif ph < max_v:
                events.append({
                    "appearance_index": n,
                    "location_index": max_x1,
                    "type": "missed_high",
                    "price": float(max_v),
                })
                events.append({
                    "appearance_index": n,
                    "location_index": follow_min_x1,
                    "type": "missed_low",
                    "price": float(follow_min),
                })

            events.append({
                "appearance_index": n,
                "location_index": p,
                "type": "regular_high",
                "price": float(ph),
            })

            # Pine reset after PH.
            os = 1
            max_v = float(ph)
            min_v = float(ph)

        if pl is not None:
            if prev_os == 0:
                events.append({
                    "appearance_index": n,
                    "location_index": max_x1,
                    "type": "missed_high",
                    "price": float(max_v),
                })

            elif pl > min_v:
                events.append({
                    "appearance_index": n,
                    "location_index": follow_max_x1,
                    "type": "missed_high",
                    "price": float(follow_max),
                })
                events.append({
                    "appearance_index": n,
                    "location_index": min_x1,
                    "type": "missed_low",
                    "price": float(min_v),
                })

            events.append({
                "appearance_index": n,
                "location_index": p,
                "type": "regular_low",
                "price": float(pl),
            })

            # Pine reset after PL.
            os = 0
            max_v = float(pl)
            min_v = float(pl)

    events.sort(key=lambda e: (e["appearance_index"], e["type"]))
    return events


def level_touched(high, low, level, tolerance=TOUCH_TOLERANCE):
    upper = level * (1 + tolerance)
    lower = level * (1 - tolerance)
    return low <= upper and high >= lower


def bullish_candle(row):
    return float(row["close"]) > float(row["open"])


def bearish_candle(row):
    return float(row["close"]) < float(row["open"])


def calculate_fee(notional=NOTIONAL):
    fee = notional * FEE_RATE
    return fee + fee * GST_ON_FEE


def apply_entry_slippage(price, side):
    return price * (1 + SLIPPAGE) if side == "LONG" else price * (1 - SLIPPAGE)


def apply_exit_slippage(price, side):
    return price * (1 - SLIPPAGE) if side == "LONG" else price * (1 + SLIPPAGE)


def calculate_gross_pnl(side, entry_price, exit_price):
    quantity = NOTIONAL / entry_price
    if side == "LONG":
        return (exit_price - entry_price) * quantity
    return (entry_price - exit_price) * quantity


def find_setup(df, events, i):
    """
    Signal uses reversal levels that had already APPEARED before
    the signal candle. We never use an event that appears later.
    """
    if i < 1:
        return None

    row = df.iloc[i]

    recent = [
        e for e in events
        if i - 10 <= e["appearance_index"] <= i
        and e["appearance_index"] <= i
    ]

    # Long: a missed/dynamic low is support.
    low_events = [
        e for e in recent
        if e["type"] == "missed_low"
        and e["price"] <= float(row["high"])
    ]

    if low_events and bullish_candle(row):
        e = low_events[-1]
        level = float(e["price"])

        if level_touched(
            float(row["high"]),
            float(row["low"]),
            level
        ):
            return {
                "side": "LONG",
                "signal_index": i,
                "level": level,
                "confirmation_high": float(row["high"]),
                "stop_loss": float(row["low"]),
            }

    # Short: a missed/dynamic high is resistance.
    high_events = [
        e for e in recent
        if e["type"] == "missed_high"
        and e["price"] >= float(row["low"])
    ]

    if high_events and bearish_candle(row):
        e = high_events[-1]
        level = float(e["price"])

        if level_touched(
            float(row["high"]),
            float(row["low"]),
            level
        ):
            return {
                "side": "SHORT",
                "signal_index": i,
                "level": level,
                "confirmation_low": float(row["low"]),
                "stop_loss": float(row["high"]),
            }

    return None


def first_dynamic_target_after_entry(events, entry_index, entry_price, side):
    """
    Return the FIRST missed reversal that APPEARS after entry.

    The appearance_index is the candle on which the reversal becomes
    known to the historical observer. The plotted location_index can
    be earlier; that historical location is NOT used as availability
    time.
    """
    for event in events:
        appearance = int(event["appearance_index"])

        if appearance < entry_index:
            continue

        if not event["type"].startswith("missed_"):
            continue

        price = float(event["price"])

        if side == "LONG" and price > entry_price:
            return event

        if side == "SHORT" and price < entry_price:
            return event

    return None


def simulate_trade(
    df,
    events,
    setup,
    entry_index
):
    side = setup["side"]

    entry_candle = df.iloc[entry_index]

    if side == "LONG":
        trigger = setup["confirmation_high"]
        if float(entry_candle["high"]) <= trigger:
            return None
    else:
        trigger = setup["confirmation_low"]
        if float(entry_candle["low"]) >= trigger:
            return None

    entry_price = apply_entry_slippage(
        float(trigger),
        side
    )
    stop_loss = float(setup["stop_loss"])

    # Find the first reversal appearance AFTER entry.
    target_event = first_dynamic_target_after_entry(
        events,
        entry_index,
        entry_price,
        side
    )

    target_created_index = (
        int(target_event["appearance_index"])
        if target_event is not None
        else None
    )

    target = (
        float(target_event["price"])
        if target_event is not None
        else None
    )

    target_created_time = (
        df["timestamp"].iloc[target_created_index]
        if target_created_index is not None
        else pd.NaT
    )

    # ------------------------------------------------------------
    # CRITICAL CHRONOLOGY:
    #
    # Before target appears, ONLY SL is active.
    # On the candle where target appears, we do NOT count a target
    # hit from that candle's earlier intrabar movement, because the
    # reversal becomes known at the close/confirmation of that bar.
    # Target checking starts on the NEXT candle.
    # ------------------------------------------------------------

    exit_index = None
    exit_price = None
    exit_reason = None

    # First manage candles from entry through the candle immediately
    # before target appearance.
    pre_target_end = (
        target_created_index
        if target_created_index is not None
        else len(df)
    )

    for j in range(entry_index, pre_target_end):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])

        if side == "LONG":
            if low <= stop_loss:
                exit_index = j
                exit_price = stop_loss
                exit_reason = "SL"
                break
        else:
            if high >= stop_loss:
                exit_index = j
                exit_price = stop_loss
                exit_reason = "SL"
                break

    if exit_index is None and target_created_index is None:
        # No target appeared before dataset end.
        exit_index = len(df) - 1
        exit_price = float(df["close"].iloc[exit_index])
        exit_reason = "END_OF_DATA_NO_TARGET"

    if exit_index is None and target_created_index is not None:
        # Target is now locked. It is immutable from this point.
        print(
            f"TARGET LOCKED | {side} | "
            f"entry={entry_price:.10g} | "
            f"target={target:.10g} | "
            f"appeared={target_created_time}"
        )

        # Target can only be hit AFTER the candle on which it appeared.
        for j in range(target_created_index + 1, len(df)):
            row = df.iloc[j]
            high = float(row["high"])
            low = float(row["low"])

            if side == "LONG":
                sl_hit = low <= stop_loss
                target_hit = high >= target

                if sl_hit:
                    exit_index = j
                    exit_price = stop_loss
                    exit_reason = "SL"
                    break

                if target_hit:
                    exit_index = j
                    exit_price = target
                    exit_reason = "TARGET"
                    break

            else:
                sl_hit = high >= stop_loss
                target_hit = low <= target

                if sl_hit:
                    exit_index = j
                    exit_price = stop_loss
                    exit_reason = "SL"
                    break

                if target_hit:
                    exit_index = j
                    exit_price = target
                    exit_reason = "TARGET"
                    break

    if exit_index is None:
        exit_index = len(df) - 1
        exit_price = float(df["close"].iloc[exit_index])
        exit_reason = "END_OF_DATA"

    exit_price_slip = apply_exit_slippage(
        exit_price,
        side
    )

    gross_pnl = calculate_gross_pnl(
        side,
        entry_price,
        exit_price_slip
    )

    total_fee = (
        calculate_fee()
        + calculate_fee()
    )

    net_pnl = gross_pnl - total_fee

    risk = abs(
        entry_price - stop_loss
    ) * (NOTIONAL / entry_price)

    reward = (
        abs(target - entry_price) * (NOTIONAL / entry_price)
        if target is not None
        else 0.0
    )

    rr = reward / risk if risk > 0 else 0.0

    result = (
        "WIN"
        if exit_reason == "TARGET"
        else "LOSS"
        if exit_reason == "SL"
        else "OPEN"
    )

    return {
        "side": side,
        "entry_index": entry_index,
        "entry_time": df["timestamp"].iloc[entry_index],
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "target_created_index": target_created_index,
        "target_created_time": target_created_time,
        "exit_index": exit_index,
        "exit_time": df["timestamp"].iloc[exit_index],
        "exit_price": exit_price_slip,
        "exit_reason": exit_reason,
        "result": result,
        "gross_pnl": gross_pnl,
        "fees": total_fee,
        "net_pnl": net_pnl,
        "risk": risk,
        "reward": reward,
        "rr": rr,
    }


def backtest(df, events):
    trades = []
    i = 0

    while i < len(df) - 2:
        setup = find_setup(df, events, i)

        if setup is None:
            i += 1
            continue

        entry_index = i + 1
        trade = simulate_trade(
            df,
            events,
            setup,
            entry_index
        )

        if trade is None:
            i += 1
            continue

        # Hard chronology validation.
        if trade["target_created_time"] is not pd.NaT and pd.notna(
            trade["target_created_time"]
        ):
            if trade["target_created_time"] < trade["entry_time"]:
                raise RuntimeError(
                    "INVALID TRADE: target appeared before entry."
                )

            if (
                trade["exit_reason"] == "TARGET"
                and trade["exit_time"] < trade["target_created_time"]
            ):
                raise RuntimeError(
                    "INVALID TRADE: target hit before target appeared."
                )

        trade["trade_no"] = len(trades) + 1
        trades.append(trade)

        # One position at a time.
        i = trade["exit_index"] + 1

    return pd.DataFrame(trades)


def make_summary(trades, pair, days):
    if trades.empty:
        return pd.DataFrame([{
            "pair": pair,
            "days": days,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
            "average_rr": 0.0,
            "max_drawdown": 0.0,
        }])

    wins = int((trades["result"] == "WIN").sum())
    losses = int((trades["result"] == "LOSS").sum())
    total = len(trades)

    win_rate = wins / total * 100

    gross_pnl = float(trades["gross_pnl"].sum())
    fees = float(trades["fees"].sum())
    net_pnl = float(trades["net_pnl"].sum())

    winning_profit = float(
        trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    )
    losing_profit = abs(float(
        trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
    ))

    profit_factor = (
        winning_profit / losing_profit
        if losing_profit > 0
        else 0.0
    )

    average_rr = float(trades["rr"].mean())

    cumulative = trades["net_pnl"].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_drawdown = float(drawdown.max())

    no_target = int(
        (trades["target"].isna()).sum()
    )

    return pd.DataFrame([{
        "pair": pair,
        "days": days,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
        "profit_factor": profit_factor,
        "average_rr": average_rr,
        "max_drawdown": max_drawdown,
        "trades_without_target": no_target,
    }])


def validate_trades(trades):
    if trades.empty:
        return

    invalid_target_time = trades[
        trades["target_created_time"].notna()
        & (
            trades["target_created_time"]
            < trades["entry_time"]
        )
    ]

    invalid_target_exit = trades[
        (trades["exit_reason"] == "TARGET")
        & (
            trades["exit_time"]
            < trades["target_created_time"]
        )
    ]

    if len(invalid_target_time):
        raise RuntimeError(
            f"Validation failed: {len(invalid_target_time)} "
            "trades have target before entry."
        )

    if len(invalid_target_exit):
        raise RuntimeError(
            f"Validation failed: {len(invalid_target_exit)} "
            "trades hit target before target appearance."
        )

    print("\nCHRONOLOGY VALIDATION: PASSED")
    print(
        "Every target appears at/after entry, and every TARGET exit "
        "occurs after target appearance."
    )


def print_results(trades, summary):
    print("\n" + "=" * 80)
    print("LUXALGO FIRST DYNAMIC REVERSAL BACKTEST")
    print("=" * 80)

    row = summary.iloc[0]

    for label, key, fmt in [
        ("Trades", "trades", "int"),
        ("Wins", "wins", "int"),
        ("Losses", "losses", "int"),
        ("Win Rate", "win_rate", "pct"),
        ("Gross P&L", "gross_pnl", "money"),
        ("Fees", "fees", "money"),
        ("Net P&L", "net_pnl", "money"),
        ("Profit Factor", "profit_factor", "float"),
        ("Average RR", "average_rr", "float"),
        ("Max Drawdown", "max_drawdown", "money"),
    ]:
        value = row[key]
        if fmt == "int":
            print(f"{label:15}: {int(value)}")
        elif fmt == "pct":
            print(f"{label:15}: {value:.2f}%")
        elif fmt == "money":
            print(f"{label:15}: ₹{value:.2f}")
        else:
            print(f"{label:15}: {value:.3f}")

    print("\n" + "-" * 80)

    if trades.empty:
        print("No trades found.")
        return

    cols = [
        "trade_no",
        "side",
        "entry_time",
        "entry_price",
        "stop_loss",
        "target",
        "target_created_time",
        "exit_time",
        "exit_price",
        "exit_reason",
        "result",
        "net_pnl",
        "rr",
    ]

    print(
        trades[cols].to_string(index=False)
    )

    print("=" * 80)


def save_results(trades, summary, events, pair):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    trades_file = os.path.join(
        RESULTS_DIR,
        f"{pair}_trades.csv"
    )
    summary_file = os.path.join(
        RESULTS_DIR,
        "summary.csv"
    )
    events_file = os.path.join(
        RESULTS_DIR,
        f"{pair}_luxalgo_events.csv"
    )

    trades.to_csv(
        trades_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    events_df = pd.DataFrame(events)

    if not events_df.empty:
        events_df.to_csv(
            events_file,
            index=False
        )

    print("\nResults saved:")
    print(trades_file)
    print(summary_file)
    print(events_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair",
        required=True
    )

    parser.add_argument(
        "--days",
        required=True,
        type=int
    )

    parser.add_argument(
        "--price-type",
        default="LAST_PRICE",
        choices=[
            "LAST_PRICE",
            "MARK_PRICE"
        ]
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("LUXALGO FIRST DYNAMIC REVERSAL BACKTEST")
    print("=" * 80)
    print("Pair        :", args.pair)
    print("Days        :", args.days)
    print("Timeframe   :", INTERVAL)
    print("Pivot       :", PIVOT_LENGTH)
    print("Price type  :", args.price_type)
    print("Margin      :", f"₹{MARGIN:.2f}")
    print("Leverage    :", f"{LEVERAGE:.1f}x")
    print("Notional    :", f"₹{NOTIONAL:.2f}")
    print("Target      :", "FIRST MISSED REVERSAL AFTER ENTRY — LOCKED")

    df = download_history(
        args.pair,
        args.days,
        args.price_type
    )

    print("\nBuilding causal LuxAlgo events...")

    events = build_luxalgo_events(
        df,
        PIVOT_LENGTH
    )

    missed = [
        e for e in events
        if e["type"].startswith("missed_")
    ]

    regular = [
        e for e in events
        if e["type"].startswith("regular_")
    ]

    print("Total events       :", len(events))
    print("Regular pivots     :", len(regular))
    print("Missed reversals   :", len(missed))

    print("\nRunning backtest...")

    trades = backtest(
        df,
        events
    )

    validate_trades(
        trades
    )

    summary = make_summary(
        trades,
        args.pair,
        args.days
    )

    print_results(
        trades,
        summary
    )

    save_results(
        trades,
        summary,
        events,
        args.pair
    )

    print("\nBACKTEST COMPLETE.")


if __name__ == "__main__":
    main()
