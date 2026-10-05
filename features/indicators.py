import pandas as pd

from common.config import load_strategy


# exponential moving average, weights recent bars more. this is the line we expect price to pull back to
def ema(close, n):
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


# simple moving average, plain avg of last n closes. used as the trend filter
def sma(close, n):
    return close.rolling(n, min_periods=n).mean()


# wilder's smoothing (the standard way atr and rsi are averaged), same as an ema with alpha = 1/n
def wilder(series, n):
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# average true range = how much the stock typically moves per bar. used to size stop/target
def atr(df, n):
    prev_close = df["Close"].shift(1)
    # true range is the biggest of: bar's high-low, or the gap from last close to this bar's high/low
    true_range = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return wilder(true_range, n)


# relative strength index, 0-100. high = strong recent buying, low = selling / pulled back
def rsi(close, n):
    delta = close.diff()
    avg_gain = wilder(delta.clip(lower=0), n)  # avg size of up moves
    avg_loss = wilder(-delta.clip(upper=0), n)  # avg size of down moves (as positive numbers)
    return 100 - 100 / (1 + avg_gain / avg_loss)


# adds all 4 indicator columns to a ticker's ohlcv df, periods come from strategy.yaml
# columns are named after the config keys (not ema20 etc) so changing a period doesnt break the detector
# first n-1 rows of each column are nan since theres not enough history yet (warmup_days covers this)
def add_indicators(df, cfg=None):
    cfg = cfg or load_strategy()["indicators"]
    out = df.copy()
    out["ema_fast"] = ema(out["Close"], cfg["ema_fast"])
    out["sma_trend"] = sma(out["Close"], cfg["sma_trend"])
    out["atr"] = atr(out, cfg["atr"])
    out["rsi"] = rsi(out["Close"], cfg["rsi"])
    return out
