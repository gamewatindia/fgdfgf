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
        print("⚠️ Generating fallback candles...")
        np.random.seed(42)
        n = 2000
        base = 5800000.0 if "BTC" in symbol else (280000.0 if "ETH" in symbol else 12500.0)
        ret = np.random.normal(0.0002, 0.012, n)
        prices = base * np.exp(np.cumsum(ret))
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
        closes = prices
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=dates)

    return df


# =====================================================================
# 2. PIVOT PARTIAL BOOKING STRATEGY (75% TP1 + 25% Runner at Breakeven)
# =====================================================================
def run_pivot_strategy(df, length=3, margin_inr=1000.0, leverage=5.0, 
                       direction="both", fee_pct=0.0005):
    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    # Pivot Point Identification
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    ph_val = np.full(n, np.nan)
    pl_val = np.full(n, np.nan)

    for i in range(2 * length, n):
        wh = highs[i - 2 * length : i + 1]
        wl = lows[i - 2 * length : i + 1]
        if (highs[i - length] == np.max(wh)) and (np.sum(wh == highs[i - length]) == 1):
            ph[i] = True
            ph_val[i] = highs[i - length]
        if (lows[i - length] == np.min(wl)) and (np.sum(wl == lows[i - length]) == 1):
            pl[i] = True
            pl_val[i] = lows[i - length]

    # Trend Guide (50 EMA)
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values

    pos_value = margin_inr * leverage
    fee_rate = fee_pct * 2

    last_resistance = np.nan
    last_support = np.nan

    pos = 0  # 1 = Long, -1 = Short
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    stage = 0  # 1 = 100% active, 2 = 75% booked & Breakeven active

    trades_pnl = []
    trade_cycles = 0
    tp1_hits = 0
    tp2_hits = 0
    be_hits = 0
    sl_hits = 0

    for i in range(2 * length, n):
        # Update Nearest Pivot Levels
        if ph[i]:
            last_resistance = ph_val[i]
        if pl[i]:
            last_support = pl_val[i]

        # ----------------- Manage Active Trade -----------------
        if pos != 0:
            if pos == 1:  # LONG TRADE
                if stage == 1:
                    # Full SL Hit (Pivot low breached)
                    if lows[i] <= sl_price:
                        pnl = pos_value * (sl_price - entry_price) / entry_price - pos_value * fee_rate
                        trades_pnl.append(pnl)
                        sl_hits += 1
                        pos = 0
                    # TP1 Hit (Nearest Resistance reached -> Book 75%)
                    elif highs[i] >= tp1_price:
                        pnl_75 = (pos_value * 0.75) * (tp1_price - entry_price) / entry_price - (pos_value * 0.75) * fee_rate
                        trades_pnl.append(pnl_75)
                        tp1_hits += 1
                        # Shift Stop-Loss to Breakeven
                        sl_price = entry_price
                        stage = 2

                elif stage == 2:
                    # Remaining 25% exits at Breakeven (Zero Loss)
                    if lows[i] <= sl_price:
                        pnl_25 = 0.0 - (pos_value * 0.25) * fee_rate
                        trades_pnl.append(pnl_25)
                        be_hits += 1
                        pos = 0
                    # Remaining 25% hits Next Pivot (TP2)
                    elif highs[i] >= tp2_price:
                        pnl_25 = (pos_value * 0.25) * (tp2_price - entry_price) / entry_price - (pos_value * 0.25) * fee_rate
                        trades_pnl.append(pnl_25)
                        tp2_hits += 1
                        pos = 0

            elif pos == -1:  # SHORT TRADE
                if stage == 1:
                    # Full SL Hit (Pivot high breached)
                    if highs[i] >= sl_price:
                        pnl = pos_value * (entry_price - sl_price) / entry_price - pos_value * fee_rate
                        trades_pnl.append(pnl)
                        sl_hits += 1
                        pos = 0
                    # TP1 Hit (Nearest Support reached -> Book 75%)
                    elif lows[i] <= tp1_price:
                        pnl_75 = (pos_value * 0.75) * (entry_price - tp1_price) / entry_price - (pos_value * 0.75) * fee_rate
                        trades_pnl.append(pnl_75)
                        tp1_hits += 1
                        # Shift Stop-Loss to Breakeven
                        sl_price = entry_price
                        stage = 2

                elif stage == 2:
                    # Remaining 25% exits at Breakeven (Zero Loss)
                    if highs[i] >= sl_price:
                        pnl_25 = 0.0 - (pos_value * 0.25) * fee_rate
                        trades_pnl.append(pnl_25)
                        be_hits += 1
                        pos = 0
                    # Remaining 25% hits Next Pivot (TP2)
                    elif lows[i] <= tp2_price:
                        pnl_25 = (pos_value * 0.25) * (entry_price - tp2_price) / entry_price - (pos_value * 0.25) * fee_rate
                        trades_pnl.append(pnl_25)
                        tp2_hits += 1
                        pos = 0

        # ----------------- Check For New Entry -----------------
        if pos == 0:
            # Long: Pivot Low Confirmed
            if pl[i] and closes[i] > ema50[i] and direction in ["both", "long_only"]:
                pivot_low = pl_val[i]
                entry = closes[i]
                sl = pivot_low * 0.997  # SL: Just 0.3% below pivot point
                risk = entry - sl
                
                if risk > 0 and (risk / entry) < 0.05:
                    # TP1: Nearest Resistance (Last Pivot High)
                    tp1 = last_resistance if (not np.isnan(last_resistance) and last_resistance > entry * 1.01) else entry + 1.8 * risk
                    # TP2: Next Pivot Extension
                    tp2 = tp1 + 1.5 * risk

                    pos = 1
                    entry_price = entry
                    sl_price = sl
                    tp1_price = tp1
                    tp2_price = tp2
                    stage = 1
                    trade_cycles += 1

            # Short: Pivot High Confirmed
            elif ph[i] and closes[i] < ema50[i] and direction in ["both", "short_only"]:
                pivot_high = ph_val[i]
                entry = closes[i]
                sl = pivot_high * 1.003  # SL: Just 0.3% above pivot point
                risk = sl - entry
                
                if risk > 0 and (risk / entry) < 0.05:
                    # TP1: Nearest Support (Last Pivot Low)
                    tp1 = last_support if (not np.isnan(last_support) and last_support < entry * 0.99) else entry - 1.8 * risk
                    # TP2: Next Pivot Extension
                    tp2 = tp1 - 1.5 * risk

                    pos = -1
                    entry_price = entry
                    sl_price = sl
                    tp1_price = tp1
                    tp2_price = tp2
                    stage = 1
                    trade_cycles += 1

    # Metrics
    pnls = np.array(trades_pnl)
    total_pnl = np.sum(pnls) if len(pnls) > 0 else 0.0
    wins = np.sum(pnls > 0)
    losses = np.sum(pnls <= 0)
    win_rate = (wins / len(pnls) * 100) if len(pnls) > 0 else 0.0

    gross_profit = pnls[pnls > 0].sum() if np.any(pnls > 0) else 0.0
    gross_loss = np.abs(pnls[pnls < 0].sum()) if np.any(pnls < 0) else 1e-9
    profit_factor = gross_profit / gross_loss

    cum_pnl = np.cumsum(pnls) if len(pnls) > 0 else np.array([0])
    cum_max = np.maximum.accumulate(cum_pnl)
    max_dd_inr = np.min(cum_pnl - cum_max) if len(pnls) > 0 else 0.0

    return {
        "Total Trades Started": trade_cycles,
        "TP1 Hits (75% Profit Booked)": tp1_hits,
        "TP2 Hits (25% Runner Target)": tp2_hits,
        "Breakeven Exits (Zero Loss)": be_hits,
        "Full Stop-Loss Hits": sl_hits,
        "Win Rate (%)": f"{win_rate:.2f}%",
        "Total Net PnL (₹)": f"₹{total_pnl:,.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Max Drawdown (₹)": f"₹{abs(max_dd_inr):,.2f}",
        "Margin Per Trade": f"₹{margin_inr}",
        "Leverage": f"{leverage}x",
        "Effective Trade Size": f"₹{pos_value:,.2f}"
    }


# =====================================================================
# 3. RUNNER
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Pivot Partial Booking System")
    parser.add_argument("--coin", default=os.getenv("COIN", "ETH"))
    parser.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "1h"))
    parser.add_argument("--period", default=os.getenv("PERIOD", "30d"))
    parser.add_argument("--amount", type=float, default=float(os.getenv("TRADE_AMOUNT", "1000")))
    parser.add_argument("--leverage", type=float, default=float(os.getenv("LEVERAGE", "5")))
    parser.add_argument("--direction", default=os.getenv("DIRECTION", "both"))
    parser.add_argument("--pivot_length", type=int, default=int(os.getenv("PIVOT_LENGTH", "3")))
    args = parser.parse_args()

    print("=" * 65)
    print(f"🎯 PIVOT STRATEGY (75% TP1 + Breakeven Runner) | {args.coin.upper()} | {args.timeframe}")
    print(f"Period: {args.period} | Margin: ₹{args.amount} | Leverage: {args.leverage}x")
    print("=" * 65)

    df = fetch_crypto_data(coin=args.coin, timeframe=args.timeframe, period=args.period)
    results = run_pivot_strategy(
        df=df,
        length=args.pivot_length,
        margin_inr=args.amount,
        leverage=args.leverage,
        direction=args.direction
    )

    for k, v in results.items():
        print(f"{k:32}: {v}")
    print("=" * 65)
