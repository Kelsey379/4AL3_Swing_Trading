import pandas as pd
from features.indicators import add_indicators

SETUP_COLS = ["setup_id", "ticker", "setup_type", "signal_time", "signal_bar_end",
              "close", "ema_fast", "sma_trend", "atr", "rsi"]


def detect_ma_pullbacks(bars, cfg, ticker, start=None): # long only so bar t only uses bars <= t
    d = add_indicators(bars, cfg["indicators"])
    p = cfg["detector"]
    k = p["pullback_lookback_bars"]
    fast, slow = d["ema_fast"], d["sma_trend"]


    trend = (d["Close"] > slow) & (fast > slow) & (slow > slow.shift(p["trend_slope_bars"]))     # uptrend = close and fast ema above slow ma, slow ma going up

    dip = (d["Low"] <= fast * (1 + p["touch_tol"])) & (d["Close"] >= slow)
    pulled_back = dip.astype(int).rolling(k, min_periods=1).max() == 1 # pullback = some bar in last k dipped to fast ema w/o closing under slow ma

    bounce = (d["Close"] > fast) & (d["Close"] > d["High"].shift(1)) # bounce = close back above fast ema and above prev bar high

    signal = trend & pulled_back & bounce
    
    if p.get("rsi_max_during_pullback") is not None: # rsi delay during the pullback
        signal &= d["rsi"].rolling(k, min_periods=1).min() <= p["rsi_max_during_pullback"]

    signal &= d[["ema_fast", "sma_trend", "atr", "rsi"]].notna().all(axis=1)  # no setups during warmup
    
    if start is not None:
        signal &= d.index >= start

    hits = d[signal]
    out = pd.DataFrame({
        "setup_id": [f"{ticker}_{ts:%Y%m%d_%H%M}" for ts in hits.index],
        "ticker": ticker,
        "setup_type": "ma_pullback",
        "signal_time": hits.index,
        "signal_bar_end": hits["bar_end"].array,  # .array keeps tz even if empty
        "close": hits["Close"].to_numpy(),
        "ema_fast": hits["ema_fast"].to_numpy(),
        "sma_trend": hits["sma_trend"].to_numpy(),
        "atr": hits["atr"].to_numpy(),
        "rsi": hits["rsi"].to_numpy(),
    }, columns=SETUP_COLS)

    # set text cols or tickers w/ 0 setups come back as floats
    return out.astype({"setup_id": "str", "ticker": "str", "setup_type": "str"}).reset_index(drop=True)
