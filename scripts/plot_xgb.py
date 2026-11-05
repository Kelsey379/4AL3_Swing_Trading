import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, precision_recall_curve

from backtest.metrics import equity_curve
from classifiers.dataset import load_dataset
from classifiers.splits import time_split, walk_forward_folds
from classifiers.xgb import training
from common.config import DATA_DIR, PROCESSED_DIR, load_yaml

# charts for the xgboost write up but need to run train_xgb first
# python -m scripts.plot_xgb --group all
# saved pngs to data/plots/


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out/name, dpi=150)
    plt.close(fig)
    print(f"saved {name}")


def plot_correlation(train, feats, out): #heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sn.heatmap(train[feats + ["label"]].corr(), annot=True, fmt=".3f", ax=ax)
    # ax.imshow(train[feats + ["label"]].corr(), cmap="coolwarm") 
    ax.set_title("Feature correlation (train)")
    save(fig, out, "feature_correlation.png")


def plot_features_by_label(train, feats, out): # lfeature vs target box plot
    rows = int(np.ceil(len(feats) / 3))  # 2x3 only fit the 6 detector feats, macro ones got cut off
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
    for ax in axes.flat[len(feats):]:
        ax.set_visible(False)
    # for ax, f in zip(axes.flat, feats):  
    #     sn.boxplot(data=train, x="label", y=f, ax=ax, showfliers=False)
    for ax, f in zip(axes.flat, feats):
        if train[f].nunique() <= 2:  # yes/no plot win rate 
            rate = train.groupby(f)["label"].agg(["mean", "size"])
            ax.bar([str(v) for v in rate.index], rate["mean"])
            ax.axhline(train["label"].mean(), color="gray", linestyle="--", label="overall")
            ax.set(xlabel=f, ylabel="win rate", title=f"win rate by {f}")
            for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
                ax.text(i, m, f"n={n}", ha="center", va="bottom")
            ax.legend()
            continue
        sn.boxplot(data=train, x="label", y=f, ax=ax, showfliers=False)
        ax.set(xticks=[0, 1], xticklabels=["Loss", "Win"], xlabel="", title=f"{f} by outcome")
    save(fig, out, "features_by_label.png")


def plot_convergence(train, feats, best, sc, out): # pr_auc per tree for last fold
    # n = int(len(train) * 0.8)  
    # tr_idx, val_idx = np.arange(n), np.arange(n, len(train))
    tr_idx, val_idx = walk_forward_folds(train, sc["n_folds"], sc["embargo_days"])[-1]
    X, y = train[feats].to_numpy(dtype=float), train["label"].to_numpy()
    params = {**best, "n_estimators": 300}  # go past best to show overfit?
    model = training(X[tr_idx], y[tr_idx], params, eval_set=[(X[tr_idx], y[tr_idx]), (X[val_idx], y[val_idx])])
    hist = model.evals_result()
    

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(hist["validation_0"]["aucpr"], label="train")
    ax.plot(hist["validation_1"]["aucpr"], label="validation (last fold)")

    ax.axhline(y[val_idx].mean(), color="gray", linestyle="--", label="base rate")
    ax.axvline(best["n_estimators"], color="red", linestyle=":", label=f"chosen ({best['n_estimators']} trees)")
    ax.set(xlabel="number of trees", ylabel="PR-AUC", title="XGBoost convergence")
    ax.legend()
    save(fig, out, "convergence.png")


def plot_folds(tuning, base_rate, out): 
    folds = [c for c in tuning.columns if c.startswith("fold_")]
    fig, ax = plt.subplots(figsize=(7, 4))



    ax.bar(range(1, len(folds) + 1), tuning.loc[0, folds].to_numpy(dtype=float))
    ax.axhline(base_rate, color="gray", linestyle="--", label="base rate")
    ax.set(title="Best setting PR-AUC per walk forward fold", xlabel="Fold", ylabel="PR-AUC",
           ylim=(0, 0.5), xticks=range(1, len(folds) + 1))
    ax.legend()
    save(fig, out, "fold_pr_auc.png")


def plot_top_settings(tuning, grid_keys, out, n=10): 
    top = tuning.head(n)
    # labels = [str(row[grid_keys].to_dict()) for _, row in top.iterrows()]  
    labels = [", ".join(f"{k}={row[k]:g}" for k in grid_keys) for _, row in top.iterrows()] 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], top["mean_pr_auc"][::-1], xerr=top["std_pr_auc"][::-1], capsize=4)
    ax.set(xlabel="Mean PR-AUC +/- SD", title=f"Top {n} tuning settings")
    save(fig, out, "top_settings.png")


def plot_confusion(test, threshold, out): #

    matrix = confusion_matrix(test["label"], (test["prob"] >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    # sn.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)  
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
    ax.set(title=f"Test confusion matrix (threshold {threshold})", xlabel="Predicted", ylabel="Actual",
           xticks=[0, 1], yticks=[0, 1], xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
    save(fig, out, "confusion_matrix.png")


def plot_pr_curve(test, out):
    precision, recall, _ = precision_recall_curve(test["label"], test["prob"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label="XGBoost")
    ax.axhline(test["label"].mean(), color="gray", linestyle="--", label="base rate (guessing)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve", ylim=(0, 1))
    ax.legend()
    save(fig, out, "pr_curve.png")


def plot_prob_hist(test, threshold, out): # wins higher probs than losses?
    fig, ax = plt.subplots(figsize=(8, 5))
    bins =np.linspace(0, 1, 21)
    # ax.hist(test.loc[test["label"] == 0, "prob"], bins=bins, alpha=0.5, density=True, label="Loss")  
    ax.hist(test.loc[test["label"] == 0, "prob"], bins=bins, alpha=0.5, label="Loss")
    ax.hist(test.loc[test["label"] == 1, "prob"], bins=bins, alpha=0.5, label="Win")
    ax.axvline(threshold, color="red", linestyle=":", label="threshold")
    ax.set(xlabel="Predicted win probability", ylabel="Setups", title="Test predictions by actual outcome")
    ax.legend()
    save(fig, out, "prob_hist.png")


def plot_importance(importance, out):
    imp= pd.Series(importance).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.to_numpy())
    ax.set(xlabel="Importance (gain share)", title="XGBoost feature importance")
    save(fig, out, "feature_importance.png")


def plot_equity(test, threshold, out): # cumul R + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, t in [("all test setups", test), ("model picks", test[test["prob"] >= threshold])]:
        eq = equity_curve(t)
        ax1.plot(eq.index, eq.to_numpy(), label=f"{name} ({len(t)} trades)")
        # dd, peak = [], 0  # loop version
        # for v in eq.to_numpy():
        #     peak = max(peak, v)
        #     dd.append(peak - v)
        dd = np.maximum.accumulate(np.r_[0.0, eq.to_numpy()])[1:] - eq.to_numpy()
        ax2.plot(eq.index, -dd, label=name)
    ax1.set(ylabel="Cumulative R", title="Test period equity curve")
    ax1.legend()
    ax2.set(ylabel="Drawdown (R)", xlabel="Exit date")
    save(fig, out, "equity_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="1, 2, 3 or all")
    args = ap.parse_args()

    mcfg = load_yaml("model.yaml")
    sc, feats, thr= mcfg["split"], mcfg["features"], mcfg["threshold"]
    with open(PROCESSED_DIR / f"xgb_results_g{args.group}.json", encoding="utf-8") as f:
        results = json.load(f)
    tuning = pd.read_csv(PROCESSED_DIR / f"xgb_tuning_g{args.group}.csv")
    preds = pd.read_parquet(PROCESSED_DIR / f"xgb_test_preds_g{args.group}.parquet")

    df = load_dataset(args.group)
    
    train_idx, _ = time_split(df, sc["test_frac"], sc["embargo_days"])  # same split as train_xgb
    train = df.iloc[train_idx].reset_index(drop=True)
    test = preds.merge(df[["setup_id", "exit_time"]], on="setup_id")

    out = DATA_DIR /"plots"
    out.mkdir(parents=True, exist_ok=True)
    plot_correlation(train, feats, out)
    plot_features_by_label(train, feats, out)
    plot_convergence(train, feats, results["best_params"], sc, out)
    plot_folds(tuning, train["label"].mean(), out)
    plot_top_settings(tuning, list(mcfg["grid"]), out)
    plot_confusion(test, thr, out)
    plot_pr_curve(test, out)
    plot_prob_hist(test, thr, out)
    plot_importance(results["importance"], out)
    plot_equity(test, thr, out)
    print(f"\nall plots in {out}")


if __name__ == "__main__":
    main()
