import time

import pandas as pd
import yfinance as yf

from common.config import DATA_DIR, load_yaml

RAW_MACRO_DIR = DATA_DIR / "raw" / "macro"


def vix_ticker():
    return load_yaml("tickers.yaml")["macro"][0]  # "^VIX"


#defined path for the cached vix data
def cache_path():
    return RAW_MACRO_DIR / "vix_1d.parquet"


#empty dataframe with the same structure as the vix data, used when no data is available
def _empty():
    return pd.DataFrame(columns=["Close"], index=pd.DatetimeIndex([], name="date"), dtype=float)


#download vix data from yahoo finance, retrying on failure. returns a dataframe with the same structure as the cached vix data
def download_vix(start, end=None, retries=3, pause=2.0):
    for attempt in range(retries):
        try:
            df = yf.download(vix_ticker(), start=start, end=end, interval="1d", auto_adjust=True,
                             progress=False, multi_level_index=False, threads=False)
        except Exception as e:
            print(f"  vix download failed ({e}), retry {attempt + 1}/{retries}")
            df = None
        if df is not None and not df.empty:
            idx = pd.DatetimeIndex(df.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            out = df[["Close"]].astype(float).copy()
            out.index = idx.normalize().astype("datetime64[ns]")
            out.index.name = "date"
            return out
        time.sleep(pause * (attempt + 1))
    print("vix has nothing from yahoo")
    return _empty()


#load cached vix data from the parquet file, or return an empty dataframe if the file does not exist
def load_cached_vix():
    path = cache_path()
    return pd.read_parquet(path) if path.exists() else _empty()


#merge cached and fresh data removing duplicates and sorting by date
def merge_vix_cache(cached, fresh):
    merged = pd.concat([cached, fresh]) if not cached.empty else fresh
    return merged[~merged.index.duplicated(keep="last")].sort_index()


#loading and updating of vix data, determines whether to use cached data or download fresh data based on the start date and the cached data's coverage, merges accordingly
def load_or_update_vix(start, now=None):
    start = pd.Timestamp(start).normalize()
    cached = load_cached_vix()
    covers = not cached.empty and cached.index.min() <= start + pd.Timedelta(days=5)
    fetch_from = cached.index.max() - pd.Timedelta(days=7) if covers else start
    fresh = download_vix(fetch_from.strftime("%Y-%m-%d"))
    merged = merge_vix_cache(cached, fresh)
    if not merged.empty:
        cache_path().parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path())
    return merged[merged.index >= start]
