import numpy as np
import pandas as pd
from data_collection.ohlcv import TZ, resample_4h


def hourly_frame(days, closes=None, start="2026-09-01"): # fake 1h bars 9:30-15:30 on weekdays, uses closes in order if given
    stamps = [d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
        for d in pd.bdate_range(start, periods=days, tz=TZ) for h in range(7)]
    idx = pd.DatetimeIndex(stamps, name="timestamp")
    c = np.asarray(closes, dtype=float) if closes is not None else np.linspace(100, 110, len(idx))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.2, "Low": np.minimum(o, c)- 0.2,
                         "Close": c, "Volume": 1000.0}, index=idx)


def bars_from(rows, start="2026-09-01"): # 4h bars from (open, high, low, close) tuples, 2 per day
    days = (len(rows) + 1) // 2
    hourly = hourly_frame(days, start=start)
    idx = resample_4h(hourly).index[:len(rows)]
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx, dtype=float)
    df["Volume"] = 1000.0
    df["bar_end"] = resample_4h(hourly)["bar_end"].iloc[:len(rows)].to_numpy()
    return df


def random_walk_bars(n_days=120, seed=1, drift=0.0008): # random 4h bars, drift > 0 trends up
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(drift, 0.006, n_days * 7)))
    return resample_4h(hourly_frame(n_days, closes))
