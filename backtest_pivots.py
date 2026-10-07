import os
import sys
import argparse
import numpy as np
import pandas as pd

# =====================================================================
# DATA FETCHING (Supports Indian INR pairs & Exchange fallbacks)
# =====================================================================
def fetch_crypto_data(coin="BTC", timeframe="1h", period="60d"):
    """
    Fetches real candlestick data in Indian Rupees (INR).
    Uses {COIN}-INR pair automatically.
    """
    symbol = f"{coin.upper().replace('-INR', '')}-INR"
    print(f"📡 Fetching data for {symbol} | Timeframe: {timeframe} | Period: {period}...")

    # Map timeframes for yfinance
    tf_map = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
    interval = tf_map.get(timeframe, "1h")

    df = None
    try:
        import yfinance as yf
        data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
        if not data.empty and len(data) > 50:
            # Flatten MultiIndex columns if returned
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            df = data[['Open', 'High', 'Low', 'Close']].dropna()
            print(f"✅ Successfully fetched {len(df)} candles for {symbol} in INR.")
    except Exception as e:
        print(f"⚠️ Yahoo Finance fetch notice: {e}")

    # Fallback to Delta Exchange India / Synthetic data if offline
    if df is None or len(df) < 50:
        print("⚠️ Live exchange unreachable. Generating fallback data for verification...")
        np.random.seed(42)
        n = 1000
        base = 5800000.0 if "BTC" in symbol else (280000.0 if "ETH" in symbol else 12500.0)
        ret = np.random.normal(0.0002, 0.015, n)
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
# PIVOT & GHOST REVERSAL ENGINE
# =====================================================================
def run_pivot_backtest(df, length=15, strategy_type="reversal", 
                       margin_inr=1000.0, leverage=5.0, direction="both", fee_pct=0.0005):
    """
    Backtests with INR Margin & Leverage.
    - margin_inr: 1000 Rs
    - leverage: e.g. 5x (Position Size = 1000 * 5 = 5,000 Rs)
    - direction: 'both', 'long_only', 'short_only'
    """
    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    max_val = 0.0
    min_val = 1e12
    follow_max = 0.0
    follow_min = 1e12
    os_state = 0

    active_ghost_h = np.nan
    active_ghost_l = np.nan
    signals = np.zeros(n)

    # 1. Indicator Calculation (No Lookahead Bias)
    for i in range(2 * length, n):
        # Pivot High confirmation at bar i (occurred at i - length)
        w_h = highs[i - 2 * length : i + 1]
        val_h = highs[i - length]
        is_ph = (val_h == np.max(w_h)) and (np.sum(w_h == val_h) == 1)

        # Pivot Low confirmation at bar i (occurred at i - length)
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

        # Signal Logic
        sig = 0
        if strategy_type == "reversal":
            if is_pl and direction in ["both", "long_only"]:
                sig = 1  # Long
            elif is_ph and direction in ["both", "short_only"]:
                sig = -1  # Short
        elif strategy_type == "breakout":
            if not np.isnan(active_ghost_h) and closes[i] > active_ghost_h and direction in ["both", "long_only"]:
                sig = 1
            elif not np.isnan(active_ghost_l) and closes[i] < active_ghost_l and direction in ["both", "short_only"]:
                sig = -1

        signals[i] = sig

    # 2. Trade Execution & Leverage PnL Tracker
    position = 0
    entry_price = 0.0
    trades_pnl = []
    trade_details = []

    pos_value = margin_inr * leverage
    round_trip_fee = pos_value * fee_pct * 2

    for i in range(1, n):
        sig = signals[i]
        if sig != 0 and sig != position:
            if position != 0:
                # Trade Exit
                if position == 1:
                    raw_ret = (closes[i] - entry_price) / entry_price
                else:
                    raw_ret = (entry_price - closes[i]) / entry_price

                # Liquidation cap (cannot lose more than ₹1000 margin)
                raw_ret = max(raw_ret, -1.0 / leverage)
                gross_pnl = pos_value * raw_ret
                net_pnl = gross_pnl - round_trip_fee
                trades_pnl.append(net_pnl)

                trade_details.append({
                    "Type": "LONG" if position == 1 else "SHORT",
                    "Entry": entry_price,
                    "Exit": closes[i],
                    "Net PnL (₹)": round(net_pnl, 2)
                })

            position = sig
            entry_price = closes[i]

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

    # Drawdown in Rupees
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
        "Effective Trade Size": f"₹{pos_value:,.2f}"
    }


# =====================================================================
# CLI / WORKFLOW RUNNER
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Pivot Backtest with INR & Leverage")
    parser.add_argument("--coin", default=os.getenv("COIN", "BTC"))
    parser.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "1h"))
    parser.add_argument("--amount", type=float, default=float(os.getenv("TRADE_AMOUNT", "1000")))
    parser.add_argument("--leverage", type=float, default=float(os.getenv("LEVERAGE", "5")))
    parser.add_argument("--direction", default=os.getenv("DIRECTION", "both"))
    parser.add_argument("--strategy", default=os.getenv("STRATEGY_TYPE", "reversal"))
    parser.add_argument("--pivot_length", type=int, default=int(os.getenv("PIVOT_LENGTH", "15")))
    args = parser.parse_args()

    print("=" * 60)
    print(f"🇮🇳 CRYPTO BACKTEST IN INR | COIN: {args.coin.upper()} | TF: {args.timeframe}")
    print(f"Margin: ₹{args.amount} | Leverage: {args.leverage}x | Direction: {args.direction.upper()}")
    print("=" * 60)

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
    print("=" * 60)
