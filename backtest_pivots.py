import numpy as np
import pandas as pd

class PivotMissedReversalBacktester:
    """
    Backtests the 'Pivot Points High Low & Missed Reversal Levels' indicator.
    Strictly prevents lookahead bias:
    - Pivots at bar [i - length] are only detected and acted upon at bar [i].
    """

    def __init__(self, df: pd.DataFrame, length: int = 15, strategy_type: str = "reversal"):
        """
        :param df: DataFrame with columns: ['Open', 'High', 'Low', 'Close']
        :param length: Symmetrical bar lookback/forward for pivots (Pine Script default is 50, 15 is recommended for trade entries)
        :param strategy_type: 'reversal' (buy on confirmed low, short on confirmed high) or 
                              'breakout' (trade breaks of missed ghost support/resistance levels)
        """
        self.df = df.copy()
        self.length = length
        self.strategy_type = strategy_type

    def detect_pivots_and_ghosts(self):
        """Replicates the Pine Script detection logic."""
        n = len(self.df)
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values

        max_val = 0.0
        min_val = 1e9
        follow_max = 0.0
        follow_min = 1e9
        os = 0

        # Arrays to hold results
        ph_signals = np.zeros(n, dtype=bool)
        pl_signals = np.zeros(n, dtype=bool)
        ghost_highs = np.full(n, np.nan)
        ghost_lows = np.full(n, np.nan)

        active_ghost_h = np.nan
        active_ghost_l = np.nan

        signals = np.zeros(n)  # 1 = Long, -1 = Short, 0 = Hold

        for i in range(2 * self.length, n):
            # Check Pivot High confirmed at bar i (peak occurred at i - length)
            w_h = highs[i - 2 * self.length : i + 1]
            val_h = highs[i - self.length]
            is_ph = (val_h == np.max(w_h)) and (np.sum(w_h == val_h) == 1)

            # Check Pivot Low confirmed at bar i (trough occurred at i - length)
            w_l = lows[i - 2 * self.length : i + 1]
            val_l = lows[i - self.length]
            is_pl = (val_l == np.min(w_l)) and (np.sum(w_l == val_l) == 1)

            curr_h = highs[i - self.length]
            curr_l = lows[i - self.length]

            prev_max, prev_min = max_val, min_val
            max_val = max(curr_h, max_val)
            min_val = min(curr_l, min_val)
            follow_max = max(curr_h, follow_max)
            follow_min = min(curr_l, follow_min)

            if max_val > prev_max:
                follow_min = curr_l
            if min_val < prev_min:
                follow_max = curr_h

            prev_os = os

            # Pivot High Event
            if is_ph:
                ph_signals[i] = True
                if prev_os == 1:
                    # Consecutive High: Missed intermediate low
                    active_ghost_l = min_val
                    ghost_lows[i] = min_val
                elif curr_h < max_val:
                    # Lower high: Missed higher peak and intermediate low
                    active_ghost_h = max_val
                    active_ghost_l = follow_min
                    ghost_highs[i] = max_val
                    ghost_lows[i] = follow_min

                os = 1
                max_val = curr_h
                min_val = curr_h

            # Pivot Low Event
            if is_pl:
                pl_signals[i] = True
                if prev_os == 0:
                    # Consecutive Low: Missed intermediate high
                    active_ghost_h = max_val
                    ghost_highs[i] = max_val
                elif curr_l > min_val:
                    # Higher low: Missed lower trough and intermediate bounce
                    active_ghost_h = follow_max
                    active_ghost_l = min_val
                    ghost_highs[i] = follow_max
                    ghost_lows[i] = min_val

                os = 0
                max_val = curr_l
                min_val = curr_l

            # Strategy Signal Assignment
            if self.strategy_type == "reversal":
                if is_pl:
                    signals[i] = 1   # Long on low confirmation
                elif is_ph:
                    signals[i] = -1  # Short on high confirmation
            elif self.strategy_type == "breakout":
                # Breakout above latest missed resistance / below missed support
                if not np.isnan(active_ghost_h) and closes[i] > active_ghost_h:
                    signals[i] = 1
                elif not np.isnan(active_ghost_l) and closes[i] < active_ghost_l:
                    signals[i] = -1

        self.df['PH'] = ph_signals
        self.df['PL'] = pl_signals
        self.df['Ghost_High'] = ghost_highs
        self.df['Ghost_Low'] = ghost_lows
        self.df['Signal'] = signals
        return self

    def run_backtest(self, initial_capital: float = 10000.0, fee_pct: float = 0.0006):
        """Executes simulation on bar closes."""
        closes = self.df['Close'].values
        signals = self.df['Signal'].values
        n = len(closes)

        position = 0  # 1 = Long, -1 = Short, 0 = Cash
        entry_price = 0.0
        portfolio_values = [initial_capital]
        trades = []

        for i in range(1, n):
            # Mark-to-market portfolio value update
            pct_change = (closes[i] - closes[i - 1]) / closes[i - 1]
            curr_val = portfolio_values[-1] * (1 + position * pct_change)

            # Signal execution
            target_signal = signals[i]
            if target_signal != 0 and target_signal != position:
                # Close existing position
                if position != 0:
                    trade_return = (
                        (closes[i] - entry_price) / entry_price
                        if position == 1
                        else (entry_price - closes[i]) / entry_price
                    )
                    trade_return -= fee_pct * 2  # Round-trip commission
                    curr_val *= (1 - fee_pct)
                    trades.append(trade_return)

                # Open new position
                position = target_signal
                entry_price = closes[i]
                curr_val *= (1 - fee_pct)

            portfolio_values.append(curr_val)

        # Performance Metrics
        portfolio_values = np.array(portfolio_values)
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
        bnh_return = (closes[-1] - closes[0]) / closes[0] * 100

        trades_arr = np.array(trades)
        num_trades = len(trades_arr)
        win_rate = (np.sum(trades_arr > 0) / num_trades * 100) if num_trades > 0 else 0

        # Maximum Drawdown
        cum_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cum_max) / cum_max
        max_drawdown = np.min(drawdowns) * 100

        # Profit Factor
        gross_profit = trades_arr[trades_arr > 0].sum() if np.any(trades_arr > 0) else 0.0
        gross_loss = np.abs(trades_arr[trades_arr < 0].sum()) if np.any(trades_arr < 0) else 1e-9
        profit_factor = gross_profit / gross_loss

        return {
            "Strategy Type": self.strategy_type.upper(),
            "Length": self.length,
            "Initial Capital ($)": initial_capital,
            "Final Equity ($)": round(portfolio_values[-1], 2),
            "Strategy Return (%)": round(total_return, 2),
            "Buy & Hold Return (%)": round(bnh_return, 2),
            "Total Trades": num_trades,
            "Win Rate (%)": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 2),
            "Max Drawdown (%)": round(max_drawdown, 2),
        }


# ==========================================================
# Example Usage & Benchmark Runner
# ==========================================================
def generate_sample_data(n_bars: int = 1500, seed: int = 42):
    """Generates synthetic OHLC prices (Geometric Brownian Motion)."""
    np.random.seed(seed)
    returns = np.random.normal(0.0003, 0.012, n_bars)
    price = 100.0 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = price * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    close = price
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    return pd.DataFrame({"Open": open_p, "High": high, "Low": low, "Close": close}, index=dates)


if __name__ == "__main__":
    # 1. Load Data:
    # Option A: Synthetic test data (Runs out of the box)
    df = generate_sample_data(n_bars=2000)

    # Option B: Use real data from CSV (Uncomment and replace filename)
    # df = pd.read_csv("BTC_1h.csv", parse_dates=True, index_col=0)

    # Option C: Use Yahoo Finance if installed (pip install yfinance)
    # import yfinance as yf
    # df = yf.download("AAPL", start="2023-01-01", interval="1d")

    print("=" * 55)
    print("BACKTEST RESULTS (Length = 15 bars)")
    print("=" * 55)

    # Model 1: Reversal Confirmation
    bt_reversal = PivotMissedReversalBacktester(df, length=15, strategy_type="reversal")
    res_reversal = bt_reversal.detect_pivots_and_ghosts().run_backtest()
    for k, v in res_reversal.items():
        print(f"{k:25}: {v}")

    print("-" * 55)

    # Model 2: Ghost Level Breakout
    bt_breakout = PivotMissedReversalBacktester(df, length=15, strategy_type="breakout")
    res_breakout = bt_breakout.detect_pivots_and_ghosts().run_backtest()
    for k, v in res_breakout.items():
        print(f"{k:25}: {v}")
    print("=" * 55)
