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

Technical features (features/technical.py, settings under technical: in strategy.yaml)
- added to every setup by build_setups, all scaled so different priced tickers compare
- macd_pct, macd_hist_pct: MACD line and histogram divided by close
- macd_hist_chg: change in the histogram over the pullback bars. Positive = momentum turning back up
- vol_long: std of log returns over the last 20 bars. vol_ratio: last 5 bars vs last 20, above 1 = vol expanding
- rel_volume: signal bar volume vs the average of the last 20 bars from the same session (morning bar is 4h, afternoon is 2.5h so they aren't mixed)
- pullback_volume: average rel_volume over the pullback bars before the signal. Low = quiet dip
- all use the signal bar and earlier only. Checked with python -m scripts.check_technical_leakage --group all

Full pipeline
1. python -m scripts.pull_ohlcv --group all --benchmarks
2. python -m scripts.build_setups --group 1 (then 2, 3, all)
3. python -m scripts.compare_groups
4. python -m scripts.check_technical_leakage --group all
5. python -m scripts.build_dataset --group all  (saves dataset_gall.parquet, same rows for every model)
6. python -m scripts.train_xgb --group all
7. python -m scripts.run_baselines --group all  (add --test-window for the xgb test dates only)
