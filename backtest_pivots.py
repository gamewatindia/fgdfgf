import os
import sys
import argparse
import numpy as np
import pandas as pd

# =====================================================================
# 1. DATA FETCHING (INR Pairs)
# =====================================================================
def fetch_crypto_data(coin="SOL", timeframe="1h", period="60d"):
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
        n = 1424
        base = 5800000.0 if "BTC" in symbol else (280000.0 if "ETH" in symbol else 12500.0)
        ret = np.random.normal(0.0002, 0.015, n)
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
# 2. FIXED STRATEGY ENGINE (Zero Over-trading + 1:2.5 Risk-Reward)
# =====================================================================
def run_pivot_backtest(df, length=10, strategy_type="reversal", 
                       margin_inr=1000.0, leverage=5.0, direction="both",
                       sl_pct=0.02, tp_pct=0.05, fee_pct=0.0005):
    """
    Fixed Model:
    - SL: 2.0%
    - TP: 5.0% (1:2.5 Risk-Reward)
    - 50 EMA Trend Filter: Only buys dips in uptrend, only sells peaks in downtrend
    - Over-trading loop strictly eliminated
    """
    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    # 50 EMA for Short-term Trend Direction
    ema50 = df['Close'].ewm(span=50, adjust=False).mean().values

    # Pivot Detection
    ph_arr = np.zeros(n, dtype=bool)
    pl_arr = np.zeros(n, dtype=bool)

    max_val = 0.0
    min_val = 1e12
    follow_max = 0.0
    follow_min = 1e12
    os_state = 0
    active_gh = np.nan
    active_gl = np.nan

    for i in range(2 * length, n):
        # Pivot High Confirmation
        w_h = highs[i - 2 * length : i + 1]
        val_h = highs[i - length]
        if (val_h == np.max(w_h)) and (np.sum(w_h == val_h) == 1):
            ph_arr[i] = True

        # Pivot Low Confirmation
        w_l = lows[i - 2 * length : i + 1]
        val_l = lows[i - length]
        if (val_l == np.min(w_l)) and (np.sum(w_l == val_l) == 1):
            pl_arr[i] = True

        # Track ghost levels
        curr_h, curr_l = highs[i - length], lows[i - length]
        prev_max, prev_min = max_val, min_val
        max_val = max(curr_h, max_val)
        min_val = min(curr_l, min_val)
        follow_max = max(curr_h, follow_max)
        follow_min = min(curr_l, follow_min)
        if max_val > prev_max: follow_min = curr_l
        if min_val < prev_min: follow_max = curr_h
        prev_os = os_state
        if ph_arr[i]:
            if prev_os == 1: active_gl = min_val
            elif curr_h < max_val: active_gh = max_val; active_gl = follow_min
            os_state = 1; max_val = curr_h; min_val = curr_h
        if pl_arr[i]:
            if prev_os == 0: active_gh = max_val
            elif curr_l > min_val: active_gh = follow_max; active_gl = min_val
            os_state = 0; max_val = curr_l; min_val = curr_l

    # Trade Execution
    pos = 0
    entry_price = 0.0
    trades_pnl = []
    
    pos_value = margin_inr * leverage
    round_trip_fee = pos_value * fee_pct * 2

    for i in range(2 * length, n):
        # Check SL / TP
        if pos != 0:
            closed = False
            pnl = 0.0

            if pos == 1:  # Long
                if lows[i] <= entry_price * (1.0 - sl_pct):
                    pnl = pos_value * (-sl_pct) - round_trip_fee
                    closed = True
                elif highs[i] >= entry_price * (1.0 + tp_pct):
                    pnl = pos_value * tp_pct - round_trip_fee
                    closed = True
            elif pos == -1:  # Short
                if highs[i] >= entry_price * (1.0 + sl_pct):
                    pnl = pos_value * (-sl_pct) - round_trip_fee
                    closed = True
                elif lows[i] <= entry_price * (1.0 - tp_pct):
                    pnl = pos_value * tp_pct - round_trip_fee
                    closed = True

            if closed:
                trades_pnl.append(pnl)
                pos = 0

        # Check for Entry ONLY when not currently in position
        if pos == 0:
            trend_up = closes[i] > ema50[i]
            trend_down = closes[i] < ema50[i]

            if strategy_type == "reversal":
                # Buy Dip in Uptrend
                if pl_arr[i] and trend_up and direction in ["both", "long_only"]:
                    pos = 1
                    entry_price = closes[i]
                # Sell Rally in Downtrend
                elif ph_arr[i] and trend_down and direction in ["both", "short_only"]:
                    pos = -1
                    entry_price = closes[i]

            elif strategy_type == "breakout":
                # Strict 1-time Crossover (No repeating loops)
                if not np.isnan(active_gh) and closes[i] > active_gh and closes[i-1] <= active_gh and trend_up:
                    pos = 1
                    entry_price = closes[i]
                    active_gh = np.nan  # Level consumed!
                elif not np.isnan(active_gl) and closes[i] < active_gl and closes[i-1] >= active_gl and trend_down:
                    pos = -1
                    entry_price = closes[i]
                    active_gl = np.nan  # Level consumed!

    # Metrics
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
        "Risk:Reward": f"SL {sl_pct*100}% | TP {tp_pct*100}% (1:2.5)"
    }


# =====================================================================
# 3. RUNNER
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Pivot Backtest Fixed")
    parser.add_argument("--coin", default=os.getenv("COIN", "SOL"))
    parser.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "1h"))
    parser.add_argument("--amount", type=float, default=float(os.getenv("TRADE_AMOUNT", "1000")))
    parser.add_argument("--leverage", type=float, default=float(os.getenv("LEVERAGE", "5")))
    parser.add_argument("--direction", default=os.getenv("DIRECTION", "both"))
    parser.add_argument("--strategy", default=os.getenv("STRATEGY_TYPE", "reversal"))
    parser.add_argument("--pivot_length", type=int, default=int(os.getenv("PIVOT_LENGTH", "10")))
    args = parser.parse_args()

    print("=" * 65)
    print(f"🎯 OPTIMIZED BACKTEST | COIN: {args.coin.upper()} | TF: {args.timeframe}")
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
