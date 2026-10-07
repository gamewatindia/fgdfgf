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
    """
    Rules:
    - Entry: Pivot confirmation
    - SL: Thode niche of Pivot Point (0.3% buffer)
    - TP1: 75% Qty booked at Nearest Resistance / Support
    - SL to Breakeven once TP1 is hit
    - TP2: 25% Qty targets Next Pivot Extension
    """
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
    stage = 0  # 1 = Initial (100%), 2 = TP1 Hit (25% remaining at Breakeven)

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
                    # Case A: Full SL Hit (Pivot low breached)
                    if lows[i] <= sl_price:
                        pnl = pos_value * (sl_price - entry_price) / entry_price - pos_value * fee_rate
                        trades_pnl.append(pnl)
                        sl_hits += 1
                        pos = 0
                    # Case B: TP1 Hit (Nearest Resistance reached -> Book 75%)
                    elif highs[i] >= tp1_price:
                        pnl_75 = (pos_value * 0.75) * (tp1_price - entry_price) / entry_price - (pos_value * 0.75) * fee_rate
                        trades_pnl.append(pnl_75)
                        tp1_hits += 1
                        # Shift Stop-Loss to Breakeven
                        sl_price = entry_price
                        stage = 2

                elif stage == 2:
                    # Case C: Remaining 25% exits at Breakeven (Risk-Free)
                    if lows[i] <= sl_price:
                        pnl_25 = 0.0 - (pos_value * 0.25) * fee_rate
                        trades_pnl.append(pnl_25)
                        be_hits += 1
                        pos = 0
                    # Case D: Remaining 25% hits Next Pivot (TP2)
                    elif highs[i] >= tp2_price:
                        pnl_25 = (pos_value * 0.25) * (tp2_price - entry_price) / entry_price - (pos_value * 0.25) * fee_rate
                        trades_pnl.append(pnl_25)
                        tp2_hits += 1
                        pos = 0

            elif pos == -1:  # SHORT TRADE
                if stage == 1:
                    # Full SL Hit
                    if highs[i] >= sl_price:
                        pnl = pos_value * (entry_price - sl_p
