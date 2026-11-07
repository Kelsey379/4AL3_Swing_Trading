import argparse

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, load_strategy
from common.leakage import check_truncation
from data_collection.ohlcv import TZ, drop_incomplete, load_cached_hourly, resample_4h
from features.technical import TECH_FEATS, build_technical_features

# leakage check for technical features, same idea as check_macro_leakage
# check_feature_times wants data strictly before bar close but tech feats use the signal bar's own close
# (same as close/rsi from the detector), entry is next bar open so <= is the right rule


def load_bars(tickers, now):
    parts = []
    for t in tickers:
        h = load_cached_hourly(t)
        if not h.empty:
            parts.append(drop_incomplete(resample_4h(h), now).assign(ticker=t))
    return pd.concat(parts) if parts else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="1", help="1, 2, 3 or all")
    ap.add_argument("--n-cuts", type=int, default=20, help="random setups to cut data right after")
    args = ap.parse_args()

    cfg = load_strategy()
    setups = pd.read_parquet(PROCESSED_DIR / f"setups_g{args.group}.parquet")
    if "tech_data_time" not in setups.columns:
        raise SystemExit("no tech cols, rerun build_setups")
    bars = load_bars(setups["ticker"].unique(), pd.Timestamp.now(tz=TZ))

    try:
        # 1. no feature uses a bar after the signal bar
        late = setups[pd.to_datetime(setups["tech_data_time"]) > setups["signal_bar_end"]]
        if len(late):
            raise ValueError(f"{len(late)} setups use bars after signal bar, e.g. {late['setup_id'].iloc[0]}")
        print(f"feature times ok: {setups['tech_data_time'].notna().sum()} rows")

        # 2. cut data off right after random setups (+ quartiles), older rows cant change
        rng = np.random.default_rng(0)
        ends = setups["signal_bar_end"].sort_values().reset_index(drop=True)
        picks = list(ends.iloc[rng.choice(len(ends), min(args.n_cuts, len(ends)), replace=False)])
        picks += [ends.iloc[int(len(ends) * q)] for q in (0.25, 0.5, 0.75)]
        cutoffs = [c + pd.Timedelta(minutes=1) for c in picks]  # +1 min so the signal bar itself is kept
        check_truncation(lambda d, s: build_technical_features(d, s, cfg), bars, setups, cutoffs, "bar_end")
        print(f"truncation test passed: {len(cutoffs)} cutoffs")

        # 3. whats saved in the parquet = fresh rebuild (catches stale setups files)
        fresh = build_technical_features(bars, setups, cfg).set_index("setup_id")[TECH_FEATS]
        saved = setups.set_index("setup_id")[TECH_FEATS]
        pd.testing.assert_frame_equal(saved, fresh, check_dtype=False)
        print("saved features match rebuild")
    except (ValueError, AssertionError) as e:
        print(f"LEAKAGE CHECK FAILED: {e}")
        raise SystemExit(1)

    print("\nNaN rate per feature:")
    print(setups[TECH_FEATS].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
