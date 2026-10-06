import time
import numpy as np
import pandas as pd
import yfinance as yf
from common.config import RAW_OHLCV_DIR

TZ = "America/New_York"
MAX_HOURLY_DAYS = 729  # can only get 730 days of 1h bars
COLS = ["Open", "High", "Low", "Close", "Volume"]

SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
BLOCK_SPLIT = pd.Timedelta(hours=13, minutes=30)
SESSION_CLOSE = pd.Timedelta(hours=16)


def cache_path(ticker, interval="1h"):
   
    return RAW_OHLCV_DIR / interval / f"{ticker}.parquet"


def _empty():
    return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz=TZ,name="timestamp"), dtype=float)




def download_hourly(ticker, start, end=None, retries=3, pause=2.0):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=True,
                             progress=False, multi_level_index=False, threads=False)
        
        except Exception as e:  # retry
            print(f"  {ticker}: download failed ({e}), retry {attempt + 1}/{retries}")
            df = None
        
        if df is not None and not df.empty:
            df= df[COLS].copy()
            df.index = df.index.tz_convert(TZ)
            df.index.name ="timestamp"
            return df
        
        time.sleep(pause * (attempt + 1))
    print(f"{ticker} has nothing from yahoo")
    return _empty()


def merge_cache(cached, fresh): # take new rows over old ones if avail
    merged = pd.concat([cached, fresh]) if not cached.empty else fresh
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def load_cached_hourly(ticker):
    path = cache_path(ticker)
    return pd.read_parquet(path) if path.exists() else _empty()


def load_or_update(ticker, days, now=None): # get only whats not in cashe for last day
    if days > MAX_HOURLY_DAYS:
        raise ValueError(f"can only pull{MAX_HOURLY_DAYS} days of data")
   
    now =now or pd.Timestamp.now(tz=TZ)
    want_start= (now - pd.Timedelta(days=days)).normalize()
    cached = load_cached_hourly(ticker)
    covers_start = not cached.empty and cached.index.min() <= want_start + pd.Timedelta(days=4) # acct for long weekends/holidays
    fetch_from= cached.index.max().normalize() if covers_start else want_start # pull from last cashed day 

    fresh = download_hourly(ticker, start=fetch_from.strftime("%Y-%m-%d"))
    merged = merge_cache(cached, fresh)
    
    if not merged.empty:
        path =cache_path(ticker)
        path.parent.mkdir(parents=True,exist_ok=True)
        merged.to_parquet(path)
    return merged[merged.index >=want_start]


def resample_4h(hourly): # get 2 4h bars per day, drop ones w/o meaning
    if hourly.empty:
        out = _empty()
        out["bar_end"] = pd.Series(dtype=f"datetime64[ns, {TZ}]")
        out.index.name = "bar_start"
        return out

    h = hourly.sort_index()
    tod = h.index - h.index.normalize()
    h = h[(tod >= SESSION_OPEN) & (tod < SESSION_CLOSE)] 
    day = h.index.normalize()
    tod = h.index - day
    block_start= day + pd.to_timedelta(np.where(tod<BLOCK_SPLIT, SESSION_OPEN.value,BLOCK_SPLIT.value))

    bars=h.groupby(block_start).agg(
        Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
        Close=("Close","last"), Volume=("Volume", "sum"),
    )
    bars.index.name = "bar_start"
    bar_tod = bars.index - bars.index.normalize()
    bars["bar_end"] = bars.index.normalize() + pd.to_timedelta(
        np.where(bar_tod == SESSION_OPEN, BLOCK_SPLIT.value, SESSION_CLOSE.value))
    return bars[bars["Volume"] > 0]


def drop_incomplete(bars, now=None): #get rid of incomplete
    now = now or pd.Timestamp.now(tz=TZ)
   
    return bars[bars["bar_end"] <= now]


def quality_report(bars, ticker):
    if bars.empty:
        return {"ticker": ticker, "n_bars": 0}
    o, h, l, c = (bars[k] for k in ["Open", "High", "Low", "Close"])
    eps = 1e-6  # acct for error (feels like 4x all over again...)
    
    bad = (h<l) |(c> h*(1 +eps)) | (c<l* (1- eps))| (o> h * (1 + eps))| (o< l* (1 -eps))
    per_day = bars.groupby(bars.index.normalize()).size()
    
    return {
        "ticker": ticker,
        "n_bars": len(bars),
        "n_days": len(per_day),
        "first": bars.index.min(),
        "last": bars.index.max(),
        "bad_ohlc": int(bad.sum()),
        "big_jumps": int((c.pct_change().abs() > 0.15).sum()),  # >15% in one bar look into 
        "one_bar_days": int((per_day < 2).sum()),  # half data/missing
    }
