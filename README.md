# LuxAlgo Pivot / Missed Reversal Backtest

GitHub-ready backtest for the LuxAlgo-style Pivot High/Low + Missed Reversal concept.

## Default strategy

- Timeframe: **5m**
- Pivot length: **50**
- Margin: **₹1,000 per trade**
- Leverage: **10x**
- LONG: support test → bullish confirmation → break of confirmation high
- SHORT: resistance test → bearish confirmation → break of confirmation low
- LONG SL: confirmation candle low
- SHORT SL: confirmation candle high
- Default target: next structural opposite level when it provides at least 1.5R
- Otherwise: 2R fallback
- Also tests fixed 1R / 1.5R / 2R / 3R
- Taker fee default: 0.040% per side
- GST on trading fee: 18%
- Slippage default: 0.02% per side

## No look-ahead

A pivot with length 50 is only made available after 50 future candles have completed. The pivot is never used at its original historical candle before confirmation.

## Data

Put OHLCV data in `data/BTCUSDT_5m.csv`:

```csv
timestamp,open,high,low,close,volume
2026-01-01T00:00:00Z,100,101,99,100.5,1234
```

The backtest intentionally takes data as a CSV so that the data source is separated from the strategy logic. This avoids accidentally mixing live trading/API credentials into the backtest.

## Run locally

```bash
pip install pandas
python luxalgo_pivot_backtest.py --data data/BTCUSDT_5m.csv --symbol BTCUSDT
```

Results:

- `results/summary.csv`
- `results/trades_structural.csv`
- `results/trades_rr_1_0.csv`
- `results/trades_rr_1_5.csv`
- `results/trades_rr_2_0.csv`
- `results/trades_rr_3_0.csv`

## GitHub Actions

The included workflow can run the backtest automatically when the CSV is updated or manually from Actions.

Important: this is a **backtest only**. It does not place live orders and does not require API keys.
