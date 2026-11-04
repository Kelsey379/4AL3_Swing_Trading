import argparse
from datetime import date, timedelta

import pandas as pd

from common.config import load_strategy
from data_collection.gdelt import pull_gdelt
from data_collection.vix import load_or_update_vix

## Pulls macro data (VIX and GDELT) from their respective sources and caches them locally.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="gdelt: only print bytes each query would scan")
    ap.add_argument("--gdelt-only", action="store_true")
    ap.add_argument("--vix-only", action="store_true")
    args = ap.parse_args()

    cfg = load_strategy()
    lookback = cfg["data"]["lookback_days"]
    chunk_days = cfg.get("macro", {}).get("gdelt_chunk_days", 30)
    today = date.today()

    if not args.gdelt_only:
        vix = load_or_update_vix(today - timedelta(days=lookback + 45))
        if len(vix):
            print(f"VIX: {len(vix)} daily rows, {vix.index.min().date()} to {vix.index.max().date()}")

    if not args.vix_only:
        start = today - timedelta(days=lookback + 7)
        end = today + timedelta(days=1)
        print(f"GDELT: {start} to {today}")
        gd = pull_gdelt(start, end, chunk_days=chunk_days, dry_run=args.dry_run)
        if gd is not None:
            print(f"GDELT: {len(gd)} hourly rows, {gd.index.min()} to {gd.index.max()}")


if __name__ == "__main__":
    main()
