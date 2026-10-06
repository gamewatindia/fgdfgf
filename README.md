# LuxAlgo Reversal Backtest

Target is now the **nearest opposite reversal level**, matching the LuxAlgo concept:

- LONG -> nearest known resistance above entry
- SHORT -> nearest known support below entry

Levels include confirmed regular pivots and the missed-reversal-style extremes tracked between confirmed pivots.

Entry:
- LONG: support test + bullish confirmation, then break of confirmation HIGH
- SHORT: resistance test + bearish confirmation, then break of confirmation LOW

SL:
- LONG = confirmation LOW
- SHORT = confirmation HIGH

Money:
- ₹1,000 margin/trade
- 10x leverage

Important: fixed 1R/1.5R/2R/3R targets are removed from the primary strategy. The target is the structural reversal level.

Pivot confirmation is delayed by 50 bars to avoid look-ahead bias.
