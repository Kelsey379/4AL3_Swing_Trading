import argparse

import pandas as pd

from common.config import PROCESSED_DIR, load_strategy
from common.leakage import check_feature_times, check_truncation
from data_collection.gdelt import load_cached_gdelt
from data_collection.vix import load_cached_vix
from features.macro_features import (MACRO_FEATS, build_gdelt_features, build_vix_features,
                                     prep_gdelt, prep_vix)


# leakage check for the macro features, using common/leakage.py:
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="1", help="1, 2, 3 or all")
    args = ap.parse_args()

    mcfg = load_strategy().get("macro", {})
    wh, rh = mcfg.get("window_hours", 72), mcfg.get("recent_hours", 24)
    setups = pd.read_parquet(PROCESSED_DIR / f"setups_g{args.group}.parquet")
    feats = pd.read_parquet(PROCESSED_DIR / f"macro_g{args.group}.parquet")
    vix, gd = prep_vix(load_cached_vix()), prep_gdelt(load_cached_gdelt())

    ends = setups["signal_bar_end"].sort_values().reset_index(drop=True)
    cutoffs = [ends.iloc[int(len(ends) * q)] for q in (0.25, 0.5, 0.75)]

    try:
        for col in ["vix_data_time", "gdelt_data_time"]:
            print(f"feature times ok for {col}: {check_feature_times(setups, feats, col)} rows")
        check_truncation(build_vix_features, vix, setups, cutoffs, "available_at")
        print("truncation test passed: vix")
        check_truncation(lambda d, s: build_gdelt_features(d, s, wh, rh), gd, setups, cutoffs, "available_at")
        print("truncation test passed: gdelt")
    except (ValueError, AssertionError) as e:
        print(f"LEAKAGE CHECK FAILED: {e}")
        raise SystemExit(1)

    print("\nNaN rate per feature:")
    print(feats[MACRO_FEATS].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
