Setup and label rules

All settings are in config/strategy.yaml. Changing them there changes them for every group.

Run
1. python -m scripts.pull_ohlcv --group 1
2. python -m scripts.build_setups --group 1
Output goes to data/processed/setups_g1.parquet. Use --group 2, 3 or all for the other groups.

Data
- 1h Yahoo bars, regular hours only, grouped into two bars per day (9:30-13:30 and 13:30-16:00).
- Bars that haven't finished yet are dropped.
- Setups are only found in the last 665 days. The 60 days before that are just warmup for the moving averages.

Setup (long only), checked at each bar using only past bars
- close above SMA50, EMA20 above SMA50, SMA50 higher than 10 bars ago
- in the last 3 bars, the low touched EMA20 (within 0.5%) without closing under SMA50
- RSI dropped to 55 or lower in those 3 bars
- close is back above EMA20 and above the previous bar's high

Label
- entry = next bar's open
- stop = entry - 1.5 x ATR, target = entry + 3 x ATR
- 10 bars max (about 5 trading days)
- win (1) if target is hit first, loss (0) if stop is hit first or time runs out
- if both are hit in the same bar, the 1h bars decide. If still unclear it counts as a loss
- label is blank if there aren't enough bars yet. Drop these before training
- only one trade per ticker at a time
