import argparse

from classifiers.dataset import load_dataset
from common.config import PROCESSED_DIR, load_yaml

# final model dataset, same file for xgb / lgbm / lstm so everyone trains on the exact same rows
# run build_setups (+ build_macro_features if using macro) first
# python -m scripts.build_dataset --group all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="1, 2, 3 or all")
    args = ap.parse_args()

    feats = load_yaml("model.yaml")["features"]
    d = load_dataset(args.group)
    missing = [f for f in feats if f not in d.columns]
    if missing:
        raise SystemExit(f"missing features {missing}. macro ones need build_macro_features, "
                         f"tech ones need build_setups rerun")

    path = PROCESSED_DIR / f"dataset_g{args.group}.parquet"
    d.to_parquet(path, index=False)
    print(f"saved {len(d)} labeled setups to {path}")
    print(f"{d['signal_time'].min():%Y-%m-%d} to {d['signal_time'].max():%Y-%m-%d}, "
          f"{d['ticker'].nunique()} tickers, {d['label'].mean():.1%} wins")

    # stuff to eyeball before training
    print("\nNaN rate per model feature (xgb ok w/ NaN, lstm will need fill):")
    print(d[feats].isna().mean().round(3).to_string())
    print("\nfeature ranges:")
    print(d[feats].describe().loc[["mean", "std", "min", "max"]].T.round(4).to_string())
    few = d["ticker"].value_counts()
    few = few[few < 5]
    if len(few):
        print(f"\nheads up, tickers w/ <5 setups: {', '.join(few.index)}")


if __name__ == "__main__":
    main()
