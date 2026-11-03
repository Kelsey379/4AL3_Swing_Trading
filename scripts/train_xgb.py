import argparse
import json
import pandas as pd

from classifiers.dataset import load_dataset
from classifiers.splits import time_split, walk_forward_folds
from classifiers.xgb import evaluate, training, tune_model
from common.config import DATA_DIR, PROCESSED_DIR, load_yaml

# run build_setups first
# python -m scripts.train_xgb --group all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="1, 2, 3 or all")
    args = ap.parse_args()

    mcfg = load_yaml("model.yaml")
    sc= mcfg["split"]
    feats = mcfg["features"]
    df = load_dataset(args.group)
    print(f"{len(df)} labeled setups, {df['label'].mean():.1%} wins, "
          f"{df['signal_time'].min():%Y-%m-%d} to {df['signal_time'].max():%Y-%m-%d}")

    train_idx, test_idx = time_split(df, sc["test_frac"], sc["embargo_days"])
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    print(f"train: {len(train)} ({train['signal_time'].min():%Y-%m-%d} to {train['signal_time'].max():%Y-%m-%d}), "
          f"test: {len(test)} ({test['signal_time'].min():%Y-%m-%d} to {test['signal_time'].max():%Y-%m-%d}), "
        f"{len(df) - len(train) - len(test)} purged")

    X_train, y_train = train[feats].to_numpy(dtype=float), train["label"].to_numpy()
    X_test, y_test = test[feats].to_numpy(dtype=float), test["label"].to_numpy()

    # tune on train only w/ walk forward folds
    folds = walk_forward_folds(train, sc["n_folds"], sc["embargo_days"])
    best, tuning = tune_model(X_train, y_train, folds, mcfg["xgb"], mcfg["grid"])
    print("\ntop 5 settings (mean pr_auc over folds):")
    print(tuning.head(5).round(3).to_string(index=False))
    print("best:", best)

    # fit on all of train, check test ONCE
    model = training(X_train, y_train, best)
    prob = model.predict_proba(X_test)[:,1]
    res = evaluate(y_test, prob, mcfg["threshold"])
    print("\ntest results:")
    for k, v in res.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    picked = test[prob >= mcfg["threshold"]]
    if len(picked):
        print(f"\navg R of test setups the model picks: {picked['ret_R'].mean():.3f} ({len(picked)} trades) "
              f"vs all test setups: {test['ret_R'].mean():.3f}")

    imp =pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nfeature importance:")
    print(imp.round(3).to_string())

    # save everything for the write up + backtest
    models_dir = DATA_DIR/"models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(models_dir / f"xgb_g{args.group}.json")
    tuning.to_csv(PROCESSED_DIR / f"xgb_tuning_g{args.group}.csv", index=False)
    test.assign(prob=prob)[["setup_id", "ticker", "signal_time", "label", "ret_R", "prob"]].to_parquet(
        PROCESSED_DIR / f"xgb_test_preds_g{args.group}.parquet", index=False)
    with open(PROCESSED_DIR / f"xgb_results_g{args.group}.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best, "test": res, "importance": imp.round(4).to_dict(),
                   "n_train": len(train), "n_test": len(test)}, f, indent=2, default=float)
    print(f"\nsaved model + results to {models_dir} and {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
