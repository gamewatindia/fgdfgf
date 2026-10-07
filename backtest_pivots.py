import os
import sys
import argparse
import numpy as np
import pandas as pd

# =====================================================================
# 1. DATA FETCHING (Supports 30d, 60d, 180d, 360d in INR)
# =====================================================================
def fetch_crypto_data(coin="ETH", timeframe="1h", period="30d"):
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

    # Fallback simulation if offline
    if df is None or len(df) < 50:
        print("⚠️ Generating fallback simulation...")
        np.random.seed(42)
        n = 3000
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
# 2. ADAPTIVE QUANT ENGINE (Auto-Scales SL/TP by Timeframe)
# =====================================================================
def run_pivot_backtest(df, timeframe="1h", length=3, margin_inr=1000.0, 
                       leverage=10.0, direction="both", fee_pct=0.0005):
    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    # Auto-adjust SL & TP to be realistic for each timeframe
    if timeframe == "15m":
        sl_pct = 0.012  # 1.2% SL
        tp_pct = 0.025  # 2.5% TP (Achievable in 15m)
        be_trigger = 0.015  # Breakeven lock at +1.5%
    elif timeframe == "1h":
        sl_pct = 0.020  # 2.0% SL
        tp_pct = 0.050  # 5.0% TP
        be_trigger = 0.025  # Breakeven lock at +2.5%
    else:
        sl_pct = 0.030
        tp_pct = 0.070
        be_trigger = 0.035

    # 1. ADX (14) for Chop/Range Elimination
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr_smooth = pd.Series(tr).rolling(14, min_periods=1).mean() + 1e-9
    plus_di = 100 * pd.Series(plus_dm).rolling(14, min_periods=1).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=1).mean() / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = np.zeros(n)
    adx[1:] = dx.rolling(14, min_periods=1).mean().values

    # 2. 50 EMA for Trend Direction
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values

    # 3. Fast Pivot Detection
    ph_arr = np.zeros(n, dtype=bool)
    pl_arr = np.zeros(n, dtype=bool)
    for i in range(2 * length, n):
        wh = highs[i - 2 * length : i + 1]
        wl = lows[i - 2 * length : i + 1]
        if (highs[i - length] == np.max(wh)) and (np.sum(wh == highs[i - length]) == 1):
            ph_arr[i] = True
        if (lows[i - length] == np.min(wl)) and (np.sum(wl == lows[i - length]) == 1):
            pl_arr[i] = True

    # 4. Trade Execution with Breakeven Protection
    pos = 0
    entry_price = 0.0
    highest_seen = 0.0
    lowest_seen = 1e12
    trades_pnl = []

    pos_value = margin_inr * leverage
    round_trip_fee = pos_value * fee_pct * 2

    for i in range(25, n):
        # Position Exit Check
        if pos != 0:
            closed = False
            pnl = 0.0

            if pos == 1:  # Long
                highest_seen = max(highest_seen, highs[i])
                # Breakeven Protection: Lock stop at entry if price reached threshold
                curr_sl = entry_price if highest_seen >= entry_price * (1.0 + be_trigger) else entry_price * (1.0 - sl_pct)
                if lows[i] <= curr_sl:
                    ret = (curr_sl - entry_price) / entry_price
                    pnl = pos_value * ret - round_trip_fee
                    closed = True
                elif highs[i] >= entry_price * (1.0 + tp_pct):
                    pnl = pos_value * tp_pct - round_trip_fee
                    closed = True

            elif pos == -1:  # Short
                lowest_seen = min(lowest_seen, lows[i])
                curr_sl = entry_price if lowest_seen <= entry_price * (1.0 - be_trigger) else entry_price * (1.0 + sl_pct)
                if highs[i] >= curr_sl:
                    ret = (entry_price - curr_sl) / entry_price
                    pnl = pos_value * ret - round_trip_fee
                    closed = True
                elif lows[i] <= entry_price * (1.0 - tp_pct):
                    pnl = pos_value * tp_pct - round_trip_fee
                    closed = True

            if closed:
                trades_pnl.append(pnl)
                pos = 0

        # Position Entry Check (ADX > 20 eliminates choppy fakeouts)
        if pos == 0 and adx[i] > 20:
            if pl_arr[i] and closes[i] > ema50[i] and direction in ["both", "long_only"]:
                pos = 1
                entry_price = closes[i]
                highest_seen = entry_price
            elif ph_arr[i] and closes[i] < ema50[i] and direction in ["both", "short_only"]:
                pos = -1
                entry_price = closes[i]
                lowest_seen = entry_price

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
        "Adaptive Risk:Reward": f"SL {sl_pct*100:.1f}% | TP {tp_pct*100:.1f}%",
        "Risk Protection": "Breakeven Lock Enabled"
    }


# =====================================================================
# 3. CLI RUNNER
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive High Profit Crypto Backtester")
    parser.add_argument("--coin", default=os.getenv("COIN", "ETH"))
    parser.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "1h"))
    parser.add_argument("--period", default=os.getenv("PERIOD", "30d"))
    parser.add_argument("--amount", type=float, default=float(os.getenv("TRADE_AMOUNT", "1000")))
    parser.add_argument("--leverage", type=float, default=float(os.getenv("LEVERAGE", "10")))
    parser.add_argument("--direction", default=os.getenv("DIRECTION", "both"))
    parser.add_argument("--pivot_length", type=int, default=int(os.getenv("PIVOT_LENGTH", "3")))
    args = parser.parse_args()

    print("=" * 65)
    print(f"💰 ADAPTIVE QUANT ENGINE | COIN: {args.coin.upper()} | TF: {args.timeframe} | PERIOD: {args.period}")
    print(f"Margin: ₹{args.amount} | Leverage: {args.leverage}x | Mode: Adaptive Timeframe")
    print("=" * 65)

    df = fetch_crypto_data(coin=args.coin, timeframe=args.timeframe, period=args.period)
    results = run_pivot_backtest(
        df=df,
        timeframe=args.timeframe,
        length=args.pivot_length,
        margin_inr=args.amount,
        leverage=args.leverage,
        direction=args.direction
    )

    for k, v in results.items():
        print(f"{k:25}: {v}")
    print("=" * 65)
