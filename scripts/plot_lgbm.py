import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from backtest.metrics import equity_curve
from classifiers.dataset import load_dataset
from classifiers.lgbm import training
from classifiers.splits import time_split, walk_forward_folds
from common.config import DATA_DIR, PROCESSED_DIR, load_yaml

# charts for the lightgbm write up, run train_lgbm first
# python -m scripts.plot_lgbm --group all
# saves pngs to data/plots/lgbm/ (own folder so the xgboost plots are never overwritten).
# the feature correlation / feature-by-outcome charts only depend on the data and split, so they are not redrawn here.


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"saved {name}")


def plot_convergence(train, feats, best, sc, out): # pr_auc per tree for last fold
    tr_idx, val_idx = walk_forward_folds(train, sc["n_folds"], sc["embargo_days"])[-1]
    X, y = train[feats].to_numpy(dtype=float), train["label"].to_numpy()
    params = {**best, "n_estimators": max(best["n_estimators"], 300)}  # go past best to show overfit
    model = training(X[tr_idx], y[tr_idx], params,
                     eval_set=[(X[tr_idx], y[tr_idx]), (X[val_idx], y[val_idx])],
                     eval_names=["train", "validation"])
    hist = model.evals_result_

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(hist["train"]["average_precision"], label="train")
    ax.plot(hist["validation"]["average_precision"], label="validation (last fold)")
    ax.axhline(y[val_idx].mean(), color="gray", linestyle="--", label="base rate")
    ax.axvline(best["n_estimators"], color="red", linestyle=":", label=f"chosen ({best['n_estimators']} trees)")
    ax.set(xlabel="number of trees", ylabel="PR-AUC", title="LightGBM convergence")
    ax.legend()
    save(fig, out, "convergence.png")


def plot_folds(tuning, base_rates, out): # each fold has its own base rate
    folds = [c for c in tuning.columns if c.startswith("fold_")]
    vals = tuning.loc[0, folds].to_numpy(dtype=float)
    x = np.arange(1, len(folds) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, vals)
    ax.hlines(base_rates, x - 0.4, x + 0.4, color="gray", linestyle="--", label="fold base rate")
    ax.set(title="Best setting PR-AUC per walk forward fold", xlabel="Fold", ylabel="PR-AUC",
           ylim=(0, max(0.5, float(np.nanmax(vals)) * 1.1)), xticks=x)
    ax.legend()
    save(fig, out, "fold_pr_auc.png")


def plot_top_settings(tuning, grid_keys, out, n=10):
    top = tuning.head(n)
    labels = [", ".join(f"{k}={row[k]:g}" for k in grid_keys) for _, row in top.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], top["mean_pr_auc"][::-1], xerr=top["std_pr_auc"][::-1], capsize=4)
    ax.set(xlabel="Mean PR-AUC +/- SD", title=f"Top {n} tuning settings (LightGBM)")
    save(fig, out, "top_settings.png")


def plot_confusion(test, threshold, out):
    matrix = confusion_matrix(test["label"], (test["prob"] >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
    ax.set(title=f"Test confusion matrix, LightGBM (threshold {threshold})", xlabel="Predicted", ylabel="Actual",
           xticks=[0, 1], yticks=[0, 1], xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
    save(fig, out, "confusion_matrix.png")


def plot_pr_curve(test, out):
    precision, recall, _ = precision_recall_curve(test["label"], test["prob"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label="LightGBM")
    ax.axhline(test["label"].mean(), color="gray", linestyle="--", label="base rate (guessing)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve", ylim=(0, 1))
    ax.legend()
    save(fig, out, "pr_curve.png")


def plot_prob_hist(test, threshold, out): # do wins get higher probs than losses?
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(test.loc[test["label"] == 0, "prob"], bins=bins, alpha=0.5, label="Loss")
    ax.hist(test.loc[test["label"] == 1, "prob"], bins=bins, alpha=0.5, label="Win")
    ax.axvline(threshold, color="red", linestyle=":", label="threshold")
    ax.set(xlabel="Model score (not a real win chance)", ylabel="Setups", title="Test predictions by actual outcome (LightGBM)")
    ax.legend()
    save(fig, out, "prob_hist.png")


def plot_importance(importance, out):
    imp = pd.Series(importance).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(imp))))  # taller once there are 20+ features
    ax.barh(imp.index, imp.to_numpy())
    ax.set(xlabel="Importance (gain share)", title="LightGBM feature importance")
    save(fig, out, "feature_importance.png")


def plot_equity(test, threshold, out): # cumul R + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, t in [("all test setups", test), ("model picks", test[test["prob"] >= threshold])]:
        eq = equity_curve(t)
        ax1.plot(eq.index, eq.to_numpy(), label=f"{name} ({len(t)} trades)")
        dd = np.maximum.accumulate(np.r_[0.0, eq.to_numpy()])[1:] - eq.to_numpy()
        ax2.plot(eq.index, -dd, label=name)
    ax1.set(ylabel="Cumulative R", title="Test period equity curve (LightGBM)")
    ax1.legend()
    ax2.set(ylabel="Drawdown (R)", xlabel="Exit date")
    save(fig, out, "equity_curve.png")


def plot_comparison(lgbm_test, xgb_test, threshold, out): # both models on the same test setups
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, t in [("LightGBM", lgbm_test), ("XGBoost", xgb_test)]:
        precision, recall, _ = precision_recall_curve(t["label"], t["prob"])
        ax1.plot(recall, precision, label=name)
        picks = t[t["prob"] >= threshold]
        eq = equity_curve(picks)
        ax2.plot(eq.index, eq.to_numpy(), label=f"{name} picks ({len(picks)} trades)")
    allq = equity_curve(lgbm_test)
    ax2.plot(allq.index, allq.to_numpy(), color="gray", linestyle="--", label=f"all setups ({len(lgbm_test)} trades)")
    ax1.axhline(lgbm_test["label"].mean(), color="gray", linestyle="--", label="base rate")
    ax1.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall: LightGBM vs XGBoost", ylim=(0, 1))
    ax2.set(xlabel="Exit date", ylabel="Cumulative R", title="Test equity: model picks vs all setups")
    ax1.legend()
    ax2.legend()
    save(fig, out, "comparison_vs_xgb.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="1, 2, 3 or all")
    args = ap.parse_args()

    mcfg = load_yaml("model.yaml")
    sc, feats, thr = mcfg["split"], mcfg["features"], mcfg["threshold"]
    with open(PROCESSED_DIR / f"lgbm_results_g{args.group}.json", encoding="utf-8") as f:
        results = json.load(f)
    tuning = pd.read_csv(PROCESSED_DIR / f"lgbm_tuning_g{args.group}.csv")
    preds = pd.read_parquet(PROCESSED_DIR / f"lgbm_test_preds_g{args.group}.parquet")

    df = load_dataset(args.group)
    train_idx, _ = time_split(df, sc["test_frac"], sc["embargo_days"])  # same split as train_lgbm / train_xgb
    train = df.iloc[train_idx].reset_index(drop=True)
    test = preds.merge(df[["setup_id", "exit_time"]], on="setup_id")

    out = DATA_DIR / "plots" / "lgbm"
    out.mkdir(parents=True, exist_ok=True)
    plot_convergence(train, feats, results["best_params"], sc, out)
    y = train["label"].to_numpy()
    plot_folds(tuning, [y[v].mean() for _, v in walk_forward_folds(train, sc["n_folds"], sc["embargo_days"])], out)
    plot_top_settings(tuning, list(mcfg["lgbm_grid"]), out)
    plot_confusion(test, thr, out)
    plot_pr_curve(test, out)
    plot_prob_hist(test, thr, out)
    plot_importance(results["importance"], out)
    plot_equity(test, thr, out)

    xgb_path = PROCESSED_DIR / f"xgb_test_preds_g{args.group}.parquet"  # optional: only if xgboost was run too
    if xgb_path.exists():
        xgb_test = pd.read_parquet(xgb_path).merge(df[["setup_id", "exit_time"]], on="setup_id")
        if set(xgb_test["setup_id"]) == set(test["setup_id"]):
            plot_comparison(test, xgb_test, thr, out)
        else:
            print("skipping comparison chart: xgboost and lightgbm were run on different test setups, "
                  "rerun both after the latest build_setups")
    print(f"\nall plots in {out}")


if __name__ == "__main__":
    main()
