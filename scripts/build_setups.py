import argparse
import pandas as pd

from common.config import PROCESSED_DIR, load_strategy, tickers_for
from data_collection.ohlcv import TZ, drop_incomplete, load_cached_hourly, resample_4h
from labeling.detector import detect_ma_pullbacks
from labeling.labeler import drop_overlapping, label_setups, summarize

# run pull_ohlcv first, this only reads the cached 1h data
# python -m scripts.build_setups --group 1


def build(tickers, cfg, now):
    window_start = now - pd.Timedelta(days=cfg["data"]["lookback_days"])
    parts= []
    
    for t in tickers:
        hourly =load_cached_hourly(t)
        if hourly.empty:
            print(f"{t} has no cached data, run scripts.pull_ohlcv first")
            continue
        bars = drop_incomplete(resample_4h(hourly), now)
        setups= detect_ma_pullbacks(bars, cfg, t, start=window_start)
        if len(setups):
            parts.append(label_setups(setups, bars, cfg, hourly=hourly))

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
  
    if cfg["labels"]["one_position_per_ticker"]:
        out = drop_overlapping(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="1", help="1, 2, 3 or all")
    args = ap.parse_args()

    cfg =load_strategy()
    now = pd.Timestamp.now(tz=TZ)
    out = build(tickers_for(args.group), cfg, now)
    if out.empty:
        print("no setups found")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR/f"setups_g{args.group}.parquet"
    out.to_parquet(path, index=False)
    print(f"saved {len(out)} setups to {path}\n")

    for k, v in summarize(out).items():
        print(f"{k}: {v}")
    print("\nper ticker:")
    print(out.groupby("ticker").agg(setups=("setup_id", "size"), wins=("label", "sum"),
        pending=("label", lambda s: s.isna().sum())).to_string())




if __name__ == "__main__":
    main()
