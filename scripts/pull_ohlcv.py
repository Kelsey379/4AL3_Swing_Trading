import argparse
import time
import pandas as pd

from common.config import PROCESSED_DIR, load_strategy, load_yaml, tickers_for
from data_collection.ohlcv import drop_incomplete, load_or_update, quality_report, resample_4h




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="1", help="1, 2, 3 or all")
    ap.add_argument("--benchmarks", action="store_true", help="also pull SPY/QQQ")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds b/t tickers")
    
    args = ap.parse_args()

    dcfg=load_strategy()["data"]
    days =dcfg["lookback_days"] + dcfg["warmup_days"]
    tickers= tickers_for(args.group)
    
    if args.benchmarks:
        tickers += load_yaml("tickers.yaml")["benchmarks"]

    print(f"pulling {len(tickers)} tickers, last {days} days of 1h bars " f"({dcfg['lookback_days']} window + {dcfg['warmup_days']} warm-up)")
    
    now = pd.Timestamp.now(tz="America/New_York")
    reports = []


    
    for t in tickers:
        hourly = load_or_update(t, days, now=now)
        bars = drop_incomplete(resample_4h(hourly), now)
        reports.append(quality_report(bars, t))
        print(f"{t} ticker; {len(hourly)} 1h bars to {len(bars)} 4h bars")
        time.sleep(args.pause)

    rep = pd.DataFrame(reports)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(PROCESSED_DIR / f"quality_g{args.group}.csv", index=False)
    
    print("\ncheck quality")
    print(rep.to_string(index=False))
    flagged = rep[(rep.get("bad_ohlc",0)> 0)| (rep.get("big_jumps", 0)> 0)| (rep["n_bars"]==0)]
    
    if len(flagged):
        print(f"\nheads up, go look at: {', '.join(flagged['ticker'])}")




if __name__ == "__main__":
    main()
