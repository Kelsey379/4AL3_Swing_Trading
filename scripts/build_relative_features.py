import argparse

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, load_strategy, tickers_for
from common.leakage import check_truncation
from data_collection.ohlcv import TZ, drop_incomplete, load_cached_hourly, resample_4h
from features.relative import BENCH, REL_FEATS, build_relative_features

# relative strength features for every setup + leakage check, saves data/processed/rel_g{group}.parquet
# classifiers.dataset joins it on setup_id like the macro file
# rank always uses every ticker (not just the group) so groups compare, + any extra tickers in the setups (sp500 experiment)
# python -m scripts.build_relative_features --group all


def load_bars(tickers, now):
    parts = []
    for t in tickers:
        h = load_cached_hourly(t)
        if not h.empty:
            parts.append(drop_incomplete(resample_4h(h), now).assign(ticker=t))
    return pd.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="1, 2, 3 or all")
    ap.add_argument("--n-cuts", type=int, default=20)
    args = ap.parse_args()

    cfg = load_strategy()
    setups = pd.read_parquet(PROCESSED_DIR / f"setups_g{args.group}.parquet")
    universe = sorted(set(tickers_for("all")) | set(setups["ticker"]))
    bars = load_bars(universe + [BENCH], pd.Timestamp.now(tz=TZ))
    out = build_relative_features(bars, setups, cfg)

    try:
        # signal bar close is ok (same as close/rsi), entry is next bar open
        late = out.merge(setups[["setup_id", "signal_bar_end"]], on="setup_id")
        late = late[pd.to_datetime(late["rel_data_time"]) > late["signal_bar_end"]]
        if len(late):
            raise ValueError(f"{len(late)} setups use bars after the signal bar")
        print(f"feature times ok: {out['rel_data_time'].notna().sum()} rows")

        rng = np.random.default_rng(0)
        ends = setups["signal_bar_end"].sort_values().reset_index(drop=True)
        cuts = [c + pd.Timedelta(minutes=1) for c in ends.iloc[rng.choice(len(ends), min(args.n_cuts, len(ends)), replace=False)]]
        check_truncation(lambda d, s: build_relative_features(d, s, cfg), bars, setups, cuts, "bar_end")
        print(f"truncation test passed: {len(cuts)} cutoffs")
    except (ValueError, AssertionError) as e:
        print(f"LEAKAGE CHECK FAILED: {e}")
        raise SystemExit(1)

    path = PROCESSED_DIR / f"rel_g{args.group}.parquet"
    out.to_parquet(path, index=False)
    print(f"{len(out)} setups -> {path}")
    print(out[REL_FEATS].describe().loc[["count", "mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
