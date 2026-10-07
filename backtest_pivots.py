import os
import sys
import argparse
import numpy as np
import pandas as pd

# =====================================================================
# 1. DATA FETCHING (Supports Indian INR pairs)
# =====================================================================
def fetch_crypto_data(coin="BTC", timeframe="1h", period="60d"):
    symbol = f"{coin.upper().replace('-INR', '')}-INR"
    print(f"📡 Fetching data for {symbol} | Timeframe: {timeframe} | Period: {period}...")

    tf_map = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
    interval = tf_map.get(timeframe, "1h")

    df = None
    try:
        import yfinance as yf
        data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
        if not data.empty and len(data) > 50:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            df = data[['Open', 'High', 'Low', 'Close']].dropna()
            print(f"✅ Successfully fetched {len(df)} candles for {symbol} in INR.")
    except Exception as e:
        print(f"⚠️ Live fetch notice: {e}")

    # Fallback simulation
    if df is None or len(df) < 50:
        print("⚠️ Generating fallback data...")
        np.random.seed(42)
        n = 1500
        base = 5800000.0 if "BTC" in symbol else (280000.0 if "ETH" in symbol else 12500.0)
        ret = np.random.normal(0.0003, 0.015, n)
        prices = base * np.exp(np.cumsum(ret))
        highs = prices * (1 + np.abs(np.random.normal(0, 0.006, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.006, n)))
        closes = prices
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=dates)

    return df


# =====================================================================
# 2. ADVANCED PIVOT ENGINE WITH DYNAMIC ATR TRAILING STOP
# =====================================================================
def run_pivot_backtest(df, length=15, strategy_type="breakout", 
                       margin_inr=1000.0, leverage=5.0, direction="both",
                       atr_sl_mult=1.8, atr_trail_mult=2.2, fee_pct=0.0005):
    """
    Upgraded with:
    1. Dynamic ATR Volatility Stop (Coin ke wicks ke hisaab se adjust hota hai)
    2. ATR Trailing Stop (Bade 10%-20% moves ko poora capture karta hai)
    3. 200 EMA Trend Alignment
    """
    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    # Calculate ATR (14) for Dynamic Risk
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros(n)
    atr[1:] = pd.Series(tr).rolling(14, min_periods=1).mean().values

    # 200 EMA for Macro Trend
    ema200 = df['Close'].ewm(span=200, adjust=False).mean().values

    max_val = 0.0
    min_val = 1e12
    follow_max = 0.0
    follow_min = 1e12
    os_state = 0

    active_ghost_h = np.nan
    active_ghost_l = np.nan
    signals = np.zeros(n)

    for i in range(2 * length, n):
        # Pivot High
        w_h = highs[i - 2 * length : i + 1]
        val_h = highs[i - length]
        is_ph = (val_h == np.max(w_h)) and (np.sum(w_h == val_h) == 1)

        # Pivot Low
        w_l = lows[i - 2 * length : i + 1]
        val_l = lows[i - length]
        is_pl = (val_l == np.min(w_l)) and (np.sum(w_l == val_l) == 1)

        curr_h, curr_l = highs[i - length], lows[i - length]
        prev_max, prev_min = max_val, min_val
        max_val = max(curr_h, max_val)
        min_val = min(curr_l, min_val)
        follow_max = max(curr_h, follow_max)
        follow_min = min(curr_l, follow_min)

        if max_val > prev_max:
            follow_min = curr_l
        if min_val < prev_min:
            follow_max = curr_h

        prev_os = os_state

        if is_ph:
            if prev_os == 1:
                active_ghost_l = min_val
            elif curr_h < max_val:
                active_ghost_h = max_val
                active_ghost_l = follow_min
            os_state = 1
            max_val = curr_h
            min_val = curr_h

        if is_pl:
            if prev_os == 0:
                active_ghost_h = max_val
            elif curr_l > min_val:
                active_ghost_h = follow_max
                active_ghost_l = min_val
            os_state = 0
            max_val = curr_l
            min_val = curr_l

        trend_up = closes[i] > ema200[i]
        trend_down = closes[i] < ema200[i]

        sig = 0
        if strategy_type == "breakout":
            if not np.isnan(active_ghost_h) and closes[i] > active_ghost_h and trend_up and direction in ["both", "long_only"]:
                sig = 1
            elif not np.isnan(active_ghost_l) and closes[i] < active_ghost_l and trend_down and direction in ["both", "short_only"]:
                sig = -1
        elif strategy_type == "reversal":
            if is_pl and trend_up and direction in ["both", "long_only"]:
                sig = 1
            elif is_ph and trend_down and direction in ["both", "short_only"]:
                sig = -1

        signals[i] = sig

    # 3. Execution with Dynamic ATR Trailing Stop
    pos = 0
    entry_price = 0.0
    highest_seen = 0.0
    lowest_seen = 1e12
    trades_pnl = []

    pos_value = margin_inr * leverage
    round_trip_fee = pos_value * fee_pct * 2

    for i in range(1, n):
        if pos != 0:
            closed = False
            raw_pnl = 0.0

            if pos == 1:  # LONG
                highest_seen = max(highest_seen, highs[i])
                # Dynamic Trailing Stop
                current_sl = max(entry_price - (atr_sl_mult * atr[i]), highest_seen - (atr_trail_mult * atr[i]))
                if lows[i] <= current_sl:
                    raw_ret = (current_sl - entry_price) / entry_price
                    raw_ret = max(raw_ret, -1.0 / leverage)
                    trades_pnl.append(pos_value * raw_ret - round_trip_fee)
                    pos = 0

            elif pos == -1:  # SHORT
                lowest_seen = min(lowest_seen, lows[i])
                # Dynamic Trailing Stop
                current_sl = min(entry_price + (atr_sl_mult * atr[i]), lowest_seen + (atr_trail_mult * atr[i]))
                if highs[i] >= current_sl:
                    raw_ret = (entry_price - current_sl) / entry_price
                    raw_ret = max(raw_ret, -1.0 / leverage)
                    trades_pnl.append(pos_value * raw_ret - round_trip_fee)
                    pos = 0

        # New entry
        if signals[i] != 0 and pos == 0:
            pos = int(signals[i])
            entry_price = closes[i]
            highest_seen = entry_price
            lowest_seen = entry_price

    # Results Calculation
    pnls = np.array(trades_pnl)
    total_trades = len(pnls)
    total_pnl = np.sum(pnls) if total_trades > 0 else 0.0
    wins = np.sum(pnls > 0)
    losses = np.sum(pnls <= 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    gross_profit = pnls[pnls > 0].sum() if np.any(pnls > 0) else 0.0
    gross_loss = np.abs(pnls[pnls < 0].sum()) if np.any(pnls < 0) else 1e-9
    profit_factor = gross_profit / gross_loss

    cum_pnl = np.cumsum(pnls) if total_trades > 0 else np.array([0])
    cum_max = np.maximum.accumulate(cum_pnl)
    max_dd_inr = np.min(cum_pnl - cum_max) if total_trades > 0 else 0.0

    return {
        "Total Trades": total_trades,
        "Winning Trades": int(wins),
        "Losing Trades": int(losses),
        "Win Rate (%)": f"{win_rate:.2f}%",
        "Total Net PnL (₹)": f"₹{total_pnl:,.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Max Drawdown (₹)": f"₹{abs(max_dd_inr):,.2f}",
        "Margin Per Trade": f"₹{margin_inr}",
        "Leverage": f"{leverage}x",
        "Trade Sizing": f"₹{pos_value:,.2f}",
        "Exit Model": "Dynamic ATR Trailing Stop (Rides Full Trend)"
    }


# =====================================================================
# 4. RUNNER
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Pivot Backtest with ATR Trailing Stop")
    parser.add_argument("--coin", default=os.getenv("COIN", "SOL"))
    parser.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "1h"))
    parser.add_argument("--amount", type=float, default=float(os.getenv("TRADE_AMOUNT", "1000")))
    parser.add_argument("--leverage", type=float, default=float(os.getenv("LEVERAGE", "5")))
    parser.add_argument("--direction", default=os.getenv("DIRECTION", "both"))
    parser.add_argument("--strategy", default=os.getenv("STRATEGY_TYPE", "breakout"))
    parser.add_argument("--pivot_length", type=int, default=int(os.getenv("PIVOT_LENGTH", "15")))
    args = parser.parse_args()

    print("=" * 65)
    print(f"🚀 UPGRADED ATR TRAILING BACKTEST | COIN: {args.coin.upper()} | TF: {args.timeframe}")
    print(f"Margin: ₹{args.amount} | Leverage: {args.leverage}x | Mode: {args.strategy.upper()}")
    print("=" * 65)

    df = fetch_crypto_data(coin=args.coin, timeframe=args.timeframe)
    results = run_pivot_backtest(
        df=df,
        length=args.pivot_length,
        strategy_type=args.strategy,
        margin_inr=args.amount,
        leverage=args.leverage,
        direction=args.direction
    )

    for k, v in results.items():
        print(f"{k:25}: {v}")
    print("=" * 65)
