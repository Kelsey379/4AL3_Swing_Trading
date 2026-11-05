import argparse

import pandas as pd

from common.config import PROCESSED_DIR, load_strategy
from data_collection.gdelt import load_cached_gdelt
from data_collection.vix import load_cached_vix
from features.macro_features import MACRO_FEATS, build_macro_features, prep_gdelt, prep_vix


# macro features for every setup of a group; data/processed/macro_g{group}.parquet
# (one row per setup_id, features only. classifiers.dataset joins it onto the setups by setup_id.)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="1", help="1, 2, 3 or all (must match a setups_g{group}.parquet)")
    ap.add_argument("--min-hours", type=float, default=66, help="min gdelt hours (of 72) for macro_ok")
    args = ap.parse_args()

    mcfg = load_strategy().get("macro", {})
    setups = pd.read_parquet(PROCESSED_DIR / f"setups_g{args.group}.parquet")
    vix, gd = load_cached_vix(), load_cached_gdelt()
    if vix.empty or gd.empty:
        raise SystemExit("macro cache is empty, run `python -m scripts.pull_macro` first")

    out = build_macro_features(prep_vix(vix), prep_gdelt(gd), setups,
                               mcfg.get("window_hours", 72), mcfg.get("recent_hours", 24))
    out["macro_ok"] = out[MACRO_FEATS].notna().all(axis=1) & (out["gdelt_hours_covered"] >= args.min_hours)

    path = PROCESSED_DIR / f"macro_g{args.group}.parquet"
    out.to_parquet(path, index=False)
    print(f"{len(out)} setups -> {path}")
    print(f"macro_ok: {int(out['macro_ok'].sum())}/{len(out)} "
          f"(rows not ok usually fall outside the pulled gdelt/vix range)")
    print(out[MACRO_FEATS].describe().loc[["count", "mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
