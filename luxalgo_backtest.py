import os
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# UTILITY
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def ms_to_datetime(ms):
    return datetime.fromtimestamp(
        ms / 1000,
        tz=timezone.utc
    )


# ============================================================
# SHARK EXCHANGE DOWNLOAD
# ============================================================

def download_history(pair, days, price_type="LAST_PRICE"):

    print("\n" + "=" * 80)
    print("SHARK EXCHANGE HISTORICAL DOWNLOAD")
    print("=" * 80)

    now = utc_now()

    requested_start = now - timedelta(days=days)

    current_end_ms = int(now.timestamp() * 1000)
    requested_start_ms = int(
        requested_start.timestamp() * 1000
    )

    all_candles = []

    page = 1

    while True:

        end_dt = ms_to_datetime(current_end_ms)

        print(
            f"\nDownloading page {page} "
            f"(through: {end_dt})"
        )

        params = {
            "priceType": price_type
        }

        payload = {
            "pair": pair,
            "interval": INTERVAL,
            "endTime": current_end_ms,
            "limit": LIMIT
        }

        try:

            response = requests.post(
                API_URL,
                params=params,
                json=payload,
                timeout=30
            )

        except Exception as e:

            raise RuntimeError(
                f"Shark API connection failed: {e}"
            )

        print(
            "HTTP STATUS:",
            response.status_code
        )

        if response.status_code not in (200, 201):

            print("\nAPI RESPONSE:")
            print(response.text[:3000])

            raise RuntimeError(
                f"Shark API returned HTTP "
                f"{response.status_code}"
            )

        try:

            raw = response.json()

        except Exception:

            print("\nCould not decode JSON.")
            print(response.text[:3000])

            raise RuntimeError(
                "Shark API response was not valid JSON."
            )

        # --------------------------------------------------------
        # FIND CANDLE ARRAY
        # --------------------------------------------------------

        candles = None

        if isinstance(raw, list):

            candles = raw

        elif isinstance(raw, dict):

            possible_keys = [
                "data",
                "result",
                "rows",
                "candles",
                "klines"
            ]

            for key in possible_keys:

                value = raw.get(key)

                if isinstance(value, list):

                    candles = value
                    break

                if isinstance(value, dict):

                    for subkey in [
                        "data",
                        "rows",
                        "candles",
                        "klines",
                        "result"
                    ]:

                        subvalue = value.get(
                            subkey
                        )

                        if isinstance(
                            subvalue,
                            list
                        ):

                            candles = subvalue
                            break

                    if candles is not None:
                        break

        if candles is None:

            print(
                "\nUNEXPECTED SHARK RESPONSE:"
            )

            print(
                str(raw)[:5000]
            )

            raise RuntimeError(
                "Historical data could not be parsed."
            )

        if len(candles) == 0:

            print(
                "No more candles returned."
            )

            break

        # --------------------------------------------------------
        # NORMALIZE
        # --------------------------------------------------------

        normalized = []

        for candle in candles:

            if not isinstance(candle, dict):
                continue

            try:

                start_time = candle.get(
                    "startTime"
                )

                if start_time is None:
                    continue

                if isinstance(
                    start_time,
                    str
                ):

                    stripped = start_time.strip()

                    if (
                        stripped
                        .replace(".", "", 1)
                        .isdigit()
                    ):

                        start_time = int(
                            float(stripped)
                        )

                    else:

                        dt = pd.to_datetime(
                            stripped,
                            utc=True,
                            errors="coerce"
                        )

                        if pd.isna(dt):
                            continue

                        start_time = int(
                            dt.timestamp() * 1000
                        )

                elif isinstance(
                    start_time,
                    (int, float)
                ):

                    start_time = int(
                        start_time
                    )

                    if start_time < 10_000_000_000:
                        start_time *= 1000

                else:

                    continue

                normalized.append({

                    "timestamp": start_time,

                    "open": float(
                        candle["open"]
                    ),

                    "high": float(
                        candle["high"]
                    ),

                    "low": float(
                        candle["low"]
                    ),

                    "close": float(
                        candle["close"]
                    ),

                    "volume": float(
                        candle.get(
                            "volume",
                            0
                        ) or 0
                    )
                })

            except Exception as e:

                print(
                    "Skipped malformed candle:",
                    str(candle)[:500],
                    "| Error:",
                    e
                )

        if not normalized:

            print(
                "\nFirst API item:"
            )

            print(
                str(candles[0])[:2000]
            )

            raise RuntimeError(
                "API returned candles but none "
                "could be normalized."
            )

        print(
            f"Page {page}: "
            f"{len(normalized)} valid candles"
        )

        all_candles.extend(
            normalized
        )

        # --------------------------------------------------------
        # PAGINATION
        # --------------------------------------------------------

        oldest_ms = min(
            c["timestamp"]
            for c in normalized
        )

        oldest_dt = ms_to_datetime(
            oldest_ms
        )

        print(
            "Oldest candle:",
            oldest_dt
        )

        if oldest_ms <= requested_start_ms:

            print(
                "\nRequested historical "
                "period reached."
            )

            break

        next_end_ms = oldest_ms - 1

        if next_end_ms >= current_end_ms:

            raise RuntimeError(
                "Pagination stopped moving backwards."
            )

        current_end_ms = next_end_ms

        page += 1

        time.sleep(0.30)

        if page > 500:

            raise RuntimeError(
                "Pagination safety limit reached."
            )

    # ------------------------------------------------------------
    # DATAFRAME
    # ------------------------------------------------------------

    if not all_candles:

        raise RuntimeError(
            "No historical candles downloaded."
        )

    df = pd.DataFrame(
        all_candles
    )

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df[
        df["timestamp"]
        >= requested_start_ms
    ].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    # ------------------------------------------------------------
    # REMOVE CURRENT OPEN CANDLE
    # ------------------------------------------------------------

    current_5m_start = pd.Timestamp.now(
        tz="UTC"
    ).floor("5min")

    df = df[
        df["timestamp"]
        < current_5m_start
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print(
        "Candles :",
        len(df)
    )

    if len(df):

        print(
            "From    :",
            df["timestamp"].iloc[0]
        )

        print(
            "To      :",
            df["timestamp"].iloc[-1]
        )

    expected = days * 24 * 12

    print(
        "Expected approx :",
        expected
    )

    coverage = (
        len(df) / expected * 100
        if expected
        else 0
    )

    print(
        f"Coverage        : "
        f"{coverage:.2f}%"
    )

    if len(df) < 200:

        raise RuntimeError(
            f"Not enough candles downloaded: "
            f"{len(df)}"
        )

    return df


# ============================================================
# PIVOT FUNCTIONS
# ============================================================

def is_confirmed_pivot_high(
    highs,
    index,
    length
):

    pivot_index = index - length

    if pivot_index < length:
        return False

    if pivot_index + length >= len(highs):
        return False

    value = highs[pivot_index]

    left = highs[
        pivot_index - length:
        pivot_index
    ]

    right = highs[
        pivot_index + 1:
        pivot_index + length + 1
    ]

    return (
        value >= max(left)
        and
        value >= max(right)
    )


def is_confirmed_pivot_low(
    lows,
    index,
    length
):

    pivot_index = index - length

    if pivot_index < length:
        return False

    if pivot_index + length >= len(lows):
        return False

    value = lows[pivot_index]

    left = lows[
        pivot_index - length:
        pivot_index
    ]

    right = lows[
        pivot_index + 1:
        pivot_index + length + 1
    ]

    return (
        value <= min(left)
        and
        value <= min(right)
    )


# ============================================================
# BUILD CONFIRMED PIVOTS
# ============================================================

def build_confirmed_pivots(
    df,
    length=PIVOT_LENGTH
):

    highs = df["high"].values
    lows = df["low"].values

    pivot_high = [None] * len(df)
    pivot_low = [None] * len(df)

    for i in range(len(df)):

        if is_confirmed_pivot_high(
            highs,
            i,
            length
        ):

            pivot_index = i - length

            pivot_high[pivot_index] = (
                float(
                    highs[pivot_index]
                )
            )

        if is_confirmed_pivot_low(
            lows,
            i,
            length
        ):

            pivot_index = i - length

            pivot_low[pivot_index] = (
                float(
                    lows[pivot_index]
                )
            )

    df = df.copy()

    df["pivot_high"] = pivot_high
    df["pivot_low"] = pivot_low

    return df


# ============================================================
# DYNAMIC REVERSAL EVENTS
#
# This is a historical, non-lookahead approximation of the
# LuxAlgo-style missed reversal / ghost structure.
# ============================================================

def build_dynamic_reversals(
    df
):

    highs = df["high"].values
    lows = df["low"].values

    events = []

    last_pivot_high = None
    last_pivot_low = None

    ghost_low = None
    ghost_high = None

    ghost_low_active = False
    ghost_high_active = False

    for i in range(len(df)):

        candle_high = float(
            highs[i]
        )

        candle_low = float(
            lows[i]
        )

        candle_close = float(
            df["close"].iloc[i]
        )

        # --------------------------------------------------------
        # CONFIRMED REGULAR PIVOTS
        # --------------------------------------------------------

        ph = df[
            "pivot_high"
        ].iloc[i]

        pl = df[
            "pivot_low"
        ].iloc[i]

        if pd.notna(ph):

            last_pivot_high = float(ph)

            events.append({

                "index": i,

                "type": "regular_high",

                "price": float(ph)

            })

        if pd.notna(pl):

            last_pivot_low = float(pl)

            events.append({

                "index": i,

                "type": "regular_low",

                "price": float(pl)

            })

        # --------------------------------------------------------
        # MISSED / DYNAMIC LOW
        #
        # Price breaks above previous pivot high.
        # The current low becomes a candidate reversal point.
        # --------------------------------------------------------

        if (
            last_pivot_high is not None
            and candle_close > last_pivot_high
        ):

            if not ghost_low_active:

                ghost_low = candle_low

                ghost_low_active = True

                events.append({

                    "index": i,

                    "type": "dynamic_low",

                    "price": ghost_low

                })

            else:

                if candle_low < ghost_low:

                    ghost_low = candle_low

                    events.append({

                        "index": i,

                        "type": "dynamic_low_shift",

                        "price": ghost_low

                    })

        # --------------------------------------------------------
        # MISSED / DYNAMIC HIGH
        #
        # Price breaks below previous pivot low.
        # Current high becomes candidate reversal point.
        # --------------------------------------------------------

        if (
            last_pivot_low is not None
            and candle_close < last_pivot_low
        ):

            if not ghost_high_active:

                ghost_high = candle_high

                ghost_high_active = True

                events.append({

                    "index": i,

                    "type": "dynamic_high",

                    "price": ghost_high

                })

            else:

                if candle_high > ghost_high:

                    ghost_high = candle_high

                    events.append({

                        "index": i,

                        "type": "dynamic_high_shift",

                        "price": ghost_high

                    })

        # --------------------------------------------------------
        # RESET DYNAMIC STRUCTURE WHEN OPPOSITE PIVOT APPEARS
        # --------------------------------------------------------

        if pd.notna(ph):

            ghost_high = None
            ghost_high_active = False

        if pd.notna(pl):

            ghost_low = None
            ghost_low_active = False

    return events


# ============================================================
# FIRST DYNAMIC TARGET
# ============================================================

def first_dynamic_target_after_entry(
    events,
    entry_index,
    entry_price,
    side
):

    candidates = []

    for event in events:

        event_index = event["index"]

        if event_index < entry_index:
            continue

        event_type = event["type"]

        price = float(
            event["price"]
        )

        # --------------------------------------------------------
        # LONG
        #
        # Target must be ABOVE entry.
        # Prefer dynamic reversal events.
        # --------------------------------------------------------

        if side == "LONG":

            if (
                "dynamic" in event_type
                and price > entry_price
            ):

                candidates.append(
                    event
                )

                break

        # --------------------------------------------------------
        # SHORT
        #
        # Target must be BELOW entry.
        # --------------------------------------------------------

        elif side == "SHORT":

            if (
                "dynamic" in event_type
                and price < entry_price
            ):

                candidates.append(
                    event
                )

                break

    if not candidates:
        return None

    return candidates[0]


# ============================================================
# TOUCH CHECK
# ============================================================

def level_touched(
    candle_high,
    candle_low,
    level,
    tolerance=TOUCH_TOLERANCE
):

    upper = level * (
        1 + tolerance
    )

    lower = level * (
        1 - tolerance
    )

    return (
        candle_low <= upper
        and
        candle_high >= lower
    )


# ============================================================
# CANDLE CONFIRMATION
# ============================================================

def bullish_candle(row):

    return (
        float(row["close"])
        > float(row["open"])
    )


def bearish_candle(row):

    return (
        float(row["close"])
        < float(row["open"])
    )


# ============================================================
# TRADE COST
# ============================================================

def calculate_fee(
    price,
    notional=NOTIONAL
):

    gross_fee = (
        notional * FEE_RATE
    )

    gst = (
        gross_fee * GST_ON_FEE
    )

    return gross_fee + gst


# ============================================================
# ENTRY / EXIT PRICE WITH SLIPPAGE
# ============================================================

def apply_entry_slippage(
    price,
    side
):

    if side == "LONG":

        return price * (
            1 + SLIPPAGE
        )

    return price * (
        1 - SLIPPAGE
    )


def apply_exit_slippage(
    price,
    side
):

    if side == "LONG":

        return price * (
            1 - SLIPPAGE
        )

    return price * (
        1 + SLIPPAGE
    )


# ============================================================
# PNL
# ============================================================

def calculate_gross_pnl(
    side,
    entry_price,
    exit_price
):

    quantity = (
        NOTIONAL / entry_price
    )

    if side == "LONG":

        return (
            exit_price
            - entry_price
        ) * quantity

    return (
        entry_price
        - exit_price
    ) * quantity


# ============================================================
# FIND ENTRY SETUPS
#
# We use already-known dynamic reversal levels.
# Entry is triggered only after confirmation candle breakout.
# ============================================================

def find_setup(
    df,
    events,
    current_index
):

    if current_index < 2:
        return None

    current = df.iloc[
        current_index
    ]

    previous = df.iloc[
        current_index - 1
    ]

    # --------------------------------------------------------
    # Search recent dynamic events.
    # --------------------------------------------------------

    recent_events = [
        e
        for e in events
        if (
            current_index - 10
            <= e["index"]
            <= current_index
        )
    ]

    if not recent_events:
        return None

    # --------------------------------------------------------
    # LONG SETUP
    #
    # Price touches dynamic low and forms bullish candle.
    # Entry is next candle breaking confirmation high.
    # --------------------------------------------------------

    bullish_levels = [
        e
        for e in recent_events
        if (
            "dynamic_low" in e["type"]
            and e["price"] <= current["high"]
        )
    ]

    if bullish_levels:

        level_event = bullish_levels[-1]

        level = float(
            level_event["price"]
        )

        if level <= current["high"]:

            if level_touched(
                current["high"],
                current["low"],
                level
            ):

                if bullish_candle(
                    current
                ):

                    return {

                        "side": "LONG",

                        "signal_index":
                            current_index,

                        "level":
                            level,

                        "confirmation_high":
                            float(
                                current["high"]
                            ),

                        "stop_loss":
                            float(
                                current["low"]
                            )
                    }

    # --------------------------------------------------------
    # SHORT SETUP
    # --------------------------------------------------------

    bearish_levels = [
        e
        for e in recent_events
        if (
            "dynamic_high" in e["type"]
            and e["price"] >= current["low"]
        )
    ]

    if bearish_levels:

        level_event = bearish_levels[-1]

        level = float(
            level_event["price"]
        )

        if level >= current["low"]:

            if level_touched(
                current["high"],
                current["low"],
                level
            ):

                if bearish_candle(
                    current
                ):

                    return {

                        "side": "SHORT",

                        "signal_index":
                            current_index,

                        "level":
                            level,

                        "confirmation_low":
                            float(
                                current["low"]
                            ),

                        "stop_loss":
                            float(
                                current["high"]
                            )
                    }

    return None


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    df,
    events
):

    trades = []

    i = 0

    while i < len(df) - 2:

        setup = find_setup(
            df,
            events,
            i
        )

        if setup is None:

            i += 1
            continue

        side = setup["side"]

        # ----------------------------------------------------
        # Entry trigger = next candle breakout
        # ----------------------------------------------------

        entry_index = i + 1

        entry_candle = df.iloc[
            entry_index
        ]

        if side == "LONG":

            trigger = float(
                setup["confirmation_high"]
            )

            if float(
                entry_candle["high"]
            ) <= trigger:

                i += 1
                continue

            raw_entry = trigger

        else:

            trigger = float(
                setup["confirmation_low"]
            )

            if float(
                entry_candle["low"]
            ) >= trigger:

                i += 1
                continue

            raw_entry = trigger

        entry_price = apply_entry_slippage(
            raw_entry,
            side
        )

        stop_loss = float(
            setup["stop_loss"]
        )

        # ----------------------------------------------------
        # FIRST DYNAMIC REVERSAL AFTER ENTRY
        #
        # Once found, target is LOCKED permanently.
        # ----------------------------------------------------

        target_event = (
            first_dynamic_target_after_entry(
                events,
                entry_index,
                entry_price,
                side
            )
        )

        if target_event is None:

            # No target available.
            # Do not manufacture one.
            i = entry_index + 1
            continue

        target_price = float(
            target_event["price"]
        )

        target_created_index = int(
            target_event["index"]
        )

        target_created_time = (
            df["timestamp"].iloc[
                target_created_index
            ]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # TARGET IS NOW LOCKED.
        #
        # We NEVER update target_price after this point.
        # ----------------------------------------------------

        locked_target = target_price

        print(
            "\nTARGET LOCKED:",
            side,
            "| Entry:",
            entry_price,
            "| Target:",
            locked_target,
            "| Created:",
            target_created_time
        )

        exit_index = None
        exit_price = None
        exit_reason = None

        # ----------------------------------------------------
        # TRADE MANAGEMENT
        # ----------------------------------------------------

        j = entry_index

        while j < len(df):

            candle = df.iloc[j]

            high = float(
                candle["high"]
            )

            low = float(
                candle["low"]
            )

            # ------------------------------------------------
            # LONG
            # ------------------------------------------------

            if side == "LONG":

                sl_hit = (
                    low <= stop_loss
                )

                target_hit = (
                    high >= locked_target
                )

                # Conservative rule:
                # If both occur on same candle,
                # assume SL happened first.
                if sl_hit:

                    exit_index = j
                    exit_price = stop_loss
                    exit_reason = "SL"
                    break

                if target_hit:

                    exit_index = j
                    exit_price = locked_target
                    exit_reason = "TARGET"
                    break

            # ------------------------------------------------
            # SHORT
            # ------------------------------------------------

            else:

                sl_hit = (
                    high >= stop_loss
                )

                target_hit = (
                    low <= locked_target
                )

                if sl_hit:

                    exit_index = j
                    exit_price = stop_loss
                    exit_reason = "SL"
                    break

                if target_hit:

                    exit_index = j
                    exit_price = locked_target
                    exit_reason = "TARGET"
                    break

            j += 1

        # ----------------------------------------------------
        # If still open at dataset end
        # ----------------------------------------------------

        if exit_index is None:

            exit_index = len(df) - 1

            exit_price = float(
                df["close"].iloc[
                    exit_index
                ]
            )

            exit_reason = "END_OF_DATA"

        exit_price_after_slippage = (
            apply_exit_slippage(
                exit_price,
                side
            )
        )

        gross_pnl = calculate_gross_pnl(
            side,
            entry_price,
            exit_price_after_slippage
        )

        entry_fee = calculate_fee(
            entry_price
        )

        exit_fee = calculate_fee(
            exit_price_after_slippage
        )

        total_fee = (
            entry_fee
            + exit_fee
        )

        net_pnl = (
            gross_pnl
            - total_fee
        )

        risk_per_trade = abs(
            entry_price
            - stop_loss
        ) * (
            NOTIONAL / entry_price
        )

        reward_per_trade = abs(
            locked_target
            - entry_price
        ) * (
            NOTIONAL / entry_price
        )

        rr = (
            reward_per_trade
            / risk_per_trade
            if risk_per_trade > 0
            else 0
        )

        result = (
            "WIN"
            if exit_reason == "TARGET"
            else
            "LOSS"
            if exit_reason == "SL"
            else
            "OPEN"
        )

        trade = {

            "trade_no":
                len(trades) + 1,

            "side":
                side,

            "signal_time":
                df["timestamp"].iloc[i],

            "entry_time":
                df["timestamp"].iloc[
                    entry_index
                ],

            "entry_price":
                entry_price,

            "stop_loss":
                stop_loss,

            "target":
                locked_target,

            "target_created_time":
                target_created_time,

            "target_created_index":
                target_created_index,

            "exit_time":
                df["timestamp"].iloc[
                    exit_index
                ],

            "exit_price":
                exit_price_after_slippage,

            "exit_reason":
                exit_reason,

            "result":
                result,

            "gross_pnl":
                gross_pnl,

            "fees":
                total_fee,

            "net_pnl":
                net_pnl,

            "risk":
                risk_per_trade,

            "reward":
                reward_per_trade,

            "rr":
                rr
        }

        trades.append(
            trade
        )

        # ----------------------------------------------------
        # Do not immediately enter another trade while
        # previous trade is active.
        # ----------------------------------------------------

        i = exit_index + 1

    return pd.DataFrame(
        trades
    )


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
    trades,
    pair,
    days
):

    if trades.empty:

        return pd.DataFrame([{

            "pair": pair,
            "days": days,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "gross_pnl": 0,
            "fees": 0,
            "net_pnl": 0,
            "profit_factor": 0,
            "average_rr": 0,
            "max_drawdown": 0

        }])

    wins = int(
        (
            trades["result"]
            == "WIN"
        ).sum()
    )

    losses = int(
        (
            trades["result"]
            == "LOSS"
        ).sum()
    )

    total_trades = len(
        trades
    )

    win_rate = (
        wins
        / total_trades
        * 100
    )

    gross_pnl = float(
        trades["gross_pnl"].sum()
    )

    fees = float(
        trades["fees"].sum()
    )

    net_pnl = float(
        trades["net_pnl"].sum()
    )

    winning_profit = float(
        trades.loc[
            trades["net_pnl"] > 0,
            "net_pnl"
        ].sum()
    )

    losing_profit = abs(
        float(
            trades.loc[
                trades["net_pnl"] < 0,
                "net_pnl"
            ].sum()
        )
    )

    if losing_profit > 0:

        profit_factor = (
            winning_profit
            / losing_profit
        )

    else:

        profit_factor = 0

    average_rr = float(
        trades["rr"].mean()
    )

    cumulative = (
        trades["net_pnl"]
        .cumsum()
    )

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        running_max
        - cumulative
    )

    max_drawdown = float(
        drawdown.max()
    )

    return pd.DataFrame([{

        "pair":
            pair,

        "days":
            days,

        "trades":
            total_trades,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            win_rate,

        "gross_pnl":
            gross_pnl,

        "fees":
            fees,

        "net_pnl":
            net_pnl,

        "profit_factor":
            profit_factor,

        "average_rr":
            average_rr,

        "max_drawdown":
            max_drawdown

    }])


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    trades,
    summary
):

    print("\n" + "=" * 80)
    print("LUXALGO FIRST DYNAMIC REVERSAL BACKTEST")
    print("=" * 80)

    row = summary.iloc[0]

    print(
        f"Trades        : "
        f"{int(row['trades'])}"
    )

    print(
        f"Wins          : "
        f"{int(row['wins'])}"
    )

    print(
        f"Losses        : "
        f"{int(row['losses'])}"
    )

    print(
        f"Win Rate      : "
        f"{row['win_rate']:.2f}%"
    )

    print(
        f"Gross P&L     : "
        f"₹{row['gross_pnl']:.2f}"
    )

    print(
        f"Fees + Slippage: "
        f"₹{row['fees']:.2f}"
    )

    print(
        f"Net P&L       : "
        f"₹{row['net_pnl']:.2f}"
    )

    print(
        f"Profit Factor : "
        f"{row['profit_factor']:.3f}"
    )

    print(
        f"Average RR    : "
        f"{row['average_rr']:.3f}"
    )

    print(
        f"Max Drawdown  : "
        f"₹{row['max_drawdown']:.2f}"
    )

    print("\n" + "-" * 80)

    if not trades.empty:

        display_columns = [

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
            "rr"

        ]

        print(
            trades[
                display_columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No trades found."
        )

    print("=" * 80)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    trades,
    summary,
    pair
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    trades_file = os.path.join(
        RESULTS_DIR,
        f"{pair}_trades.csv"
    )

    summary_file = os.path.join(
        RESULTS_DIR,
        "summary.csv"
    )

    trades.to_csv(
        trades_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        "\nResults saved:"
    )

    print(
        trades_file
    )

    print(
        summary_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "LuxAlgo-style First Dynamic "
            "Reversal Backtest"
        )
    )

    parser.add_argument(
        "--pair",
        required=True,
        type=str
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

    print("\n")
    print("=" * 80)
    print(
        "LUXALGO FIRST DYNAMIC "
        "REVERSAL BACKTEST"
    )
    print("=" * 80)

    print(
        f"Pair        : {args.pair}"
    )

    print(
        f"Days        : {args.days}"
    )

    print(
        f"Timeframe   : {INTERVAL}"
    )

    print(
        f"Price Type  : {args.price_type}"
    )

    print(
        f"Pivot Length: {PIVOT_LENGTH}"
    )

    print(
        f"Margin      : ₹{MARGIN}"
    )

    print(
        f"Leverage    : {LEVERAGE}x"
    )

    print(
        f"Notional    : ₹{NOTIONAL}"
    )

    print(
        "Target      : FIRST DYNAMIC "
        "REVERSAL — LOCKED"
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    df = download_history(
        pair=args.pair,
        days=args.days,
        price_type=args.price_type
    )

    # --------------------------------------------------------
    # PIVOTS
    # --------------------------------------------------------

    print("\nBuilding confirmed pivots...")

    df = build_confirmed_pivots(
        df,
        PIVOT_LENGTH
    )

    pivot_high_count = int(
        df["pivot_high"]
        .notna()
        .sum()
    )

    pivot_low_count = int(
        df["pivot_low"]
        .notna()
        .sum()
    )

    print(
        "Confirmed Pivot Highs:",
        pivot_high_count
    )

    print(
        "Confirmed Pivot Lows :",
        pivot_low_count
    )

    # --------------------------------------------------------
    # DYNAMIC REVERSALS
    # --------------------------------------------------------

    print(
        "\nBuilding dynamic reversal events..."
    )

    events = build_dynamic_reversals(
        df
    )

    dynamic_events = [
        e
        for e in events
        if "dynamic" in e["type"]
    ]

    print(
        "Total structural events:",
        len(events)
    )

    print(
        "Dynamic reversal events:",
        len(dynamic_events)
    )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    print(
        "\nRunning backtest..."
    )

    trades = backtest(
        df,
        events
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = make_summary(
        trades,
        args.pair,
        args.days
    )

    print_results(
        trades,
        summary
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        trades,
        summary,
        args.pair
    )

    print(
        "\nBACKTEST COMPLETE."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
