from datetime import timedelta

import pandas as pd

from common.clients import bq_client, bq_job_config
from common.config import DATA_DIR, load_sources

RAW_GDELT_DIR = DATA_DIR / "raw" / "gdelt"
SUM_COLS = ["n_events", "mentions", "tone_x_m", "gold_x_m", "us_mentions", "us_tone_x_m"]


#cached path for the gdelt data
def cache_path():
    return RAW_GDELT_DIR / "events_hourly.parquet"


#empty dataframe with the same structure as the gdelt data, used when no data is available
def _empty():
    return pd.DataFrame(columns=SUM_COLS, index=pd.DatetimeIndex([], tz="UTC", name="hour_utc"), dtype=float)


#build a query to fetch gdelt data from BigQuery
def build_query(table, start, end):
    return f"""
        SELECT
          TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING)), HOUR) AS hour_utc,
          COUNT(*) AS n_events,
          SUM(NumMentions) AS mentions,
          SUM(AvgTone * NumMentions) AS tone_x_m,
          SUM(GoldsteinScale * NumMentions) AS gold_x_m,
          SUM(IF(ActionGeo_CountryCode = 'US', NumMentions, 0)) AS us_mentions,
          SUM(IF(ActionGeo_CountryCode = 'US', AvgTone * NumMentions, 0)) AS us_tone_x_m
        FROM `{table}`
        WHERE _PARTITIONTIME >= TIMESTAMP('{start}')
          AND _PARTITIONTIME < TIMESTAMP('{end}')
          AND NumMentions > 0 AND AvgTone IS NOT NULL AND GoldsteinScale IS NOT NULL
        GROUP BY hour_utc
        ORDER BY hour_utc
    """


#generate date chunks for fetching gdelt data
def date_chunks(start, end, days):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


#adjusts dataframe index for correct format
def _fix_index(df):
    df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns, UTC]")
    df.index.name = "hour_utc"
    return df


#load cached data if available, otherwise return an empty dataframe
def load_cached_gdelt():
    path = cache_path()
    return _fix_index(pd.read_parquet(path)) if path.exists() else _empty()


#merges cached and fresh gdelt data, removing duplicates and sorting by index
def merge_gdelt_cache(cached, fresh):
    merged = pd.concat([cached, fresh]) if not cached.empty else fresh
    return merged[~merged.index.duplicated(keep="last")].sort_index()


#data retrieval from BigQuery, with optional dry run to estimate data size without downloading
def pull_gdelt(start, end, chunk_days=30, dry_run=False):
    table = load_sources()["gdelt"]["events_table"]
    client = bq_client()

    cached = load_cached_gdelt()
    if not cached.empty and cached.index.min() <= pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=2):
        start = max(start, cached.index.max().date())  # re-pull the last cached day, it may be partial

    parts, total_bytes = [], 0
    for a, b in date_chunks(start, end, chunk_days):
        query = build_query(table, a, b)
        dry = client.query(query, job_config=bq_job_config(dry_run=True))
        total_bytes += dry.total_bytes_processed
        print(f"  {a} to {b}: would scan {dry.total_bytes_processed / 1e9:.2f} GB")
        if dry_run:
            continue
        df = client.query(query, job_config=bq_job_config()).to_dataframe()
        if df.empty:
            continue
        df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
        df = _fix_index(df.set_index("hour_utc")[SUM_COLS].astype(float))
        parts.append(df)

    print(f"total scanned: {total_bytes / 1e9:.2f} GB" + (" (dry run, nothing downloaded)" if dry_run else ""))
    if dry_run or not parts:
        return None

    merged = merge_gdelt_cache(cached, pd.concat(parts))
    cache_path().parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path())
    return merged
