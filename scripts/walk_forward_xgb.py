import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backtest.metrics import equity_curve, summarize
from classifiers.dataset import load_dataset
from classifiers.xgb import evaluate, walk_forward_predict
from common.config import DATA_DIR, PROCESSED_DIR, load_yaml, tickers_for
from features.macro_features import MACRO_FEATS
from features.relative import REL_FEATS
from features.technical import TECH_FEATS

# walk forward eval: retrain before each time window, so ~1000 out of sample preds instead of the 266 in the test set
# use this to compare feature sets, not the test set
# python -m scripts.walk_forward_xgb --group all            <- features in model.yaml
# python -m scripts.walk_forward_xgb --group all --compare  <- every feature set side by side (slow, ~4 min each)
# python -m scripts.walk_forward_xgb --group all --sets detector,det+rel   <- just some


def feature_sets(feats, compare, sets=None):
    if not compare and not sets:
        return {"model.yaml": feats}
    det = [f for f in feats if f not in TECH_FEATS + MACRO_FEATS + REL_FEATS]
    pick = lambda group: [f for f in feats if f in group]
    every = {"detector": det, "det+tech": det + pick(TECH_FEATS), "det+macro": det + pick(MACRO_FEATS),
             "det+rel": det + pick(REL_FEATS), "all": feats}
    every = {k: v for k, v in every.items() if k == "detector" or len(v) > len(det)}  # skip sets not in model.yaml
    return {k: every[k] for k in sets} if sets else every


def roc_ci(y, p, n=2000, seed=0): # bootstrap by row, setups close in time overlap so real CI is a bit wider
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out/name, dpi=150)
    plt.close(fig)
    print(f"saved {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="1, 2, 3 or all")
    ap.add_argument("--compare", action="store_true", help="run each feature set")
    ap.add_argument("--sets", default="", help="comma list, e.g. detector,det+rel,all")
    args = ap.parse_args()

    mcfg = load_yaml("model.yaml")
    wf, thr = mcfg["walk_forward"], mcfg["threshold"]
    sets = [s for s in args.sets.split(",") if s]
    df = load_dataset(args.group)
    info = df[["setup_id", "ticker", "signal_time", "exit_time", "label", "ret_R"]]

    preds, summary, windows = [], [], []
    for name, feats in feature_sets(mcfg["features"], args.compare, sets).items():
        print(f"\n{name}: {len(feats)} features")
        p = walk_forward_predict(df, feats, mcfg, wf["n_windows"], wf["inner_folds"], wf["min_train"]).merge(info, on="setup_id")
        p["features"] = name
        preds.append(p)

        y, prob = p["label"].to_numpy(), p["prob"].to_numpy()
        res = evaluate(y, prob, thr)
        lo, hi = roc_ci(y, prob)
        picks = p[p["prob"] >= thr]
        row = {"features": name, "n_feats": len(feats), **res, "roc_lo": lo, "roc_hi": hi,
               "avg_R_all": p["ret_R"].mean(), "avg_R_picks": picks["ret_R"].mean(),
               **{f"picks_{k}": v for k, v in summarize(picks).items()}}
        for g in (1, 2, 3):
            pg = picks[picks["ticker"].isin(tickers_for(g))]
            row[f"g{g}_picks_avg_R"] = pg["ret_R"].mean() if len(pg) else np.nan
        summary.append(row)
        print(f"  {len(p)} setups predicted, {p['signal_time'].min():%Y-%m-%d} to {p['signal_time'].max():%Y-%m-%d}")
        print(f"  roc_auc {res['roc_auc']:.3f} (CI {lo:.2f} to {hi:.2f}), pr_auc {res['pr_auc']:.3f} vs base {res['base_rate']:.3f}")
        print(f"  avg R picks {row['avg_R_picks']:.3f} ({len(picks)}) vs all {row['avg_R_all']:.3f}")

        for k, w in p.groupby("window"):
            yw = w["label"].to_numpy()
            windows.append({"features": name, "window": k, "start": w["signal_time"].min(), "n": len(w),
                            "n_train": int(w["n_train"].iloc[0]), "base_rate": yw.mean(),
                            "roc_auc": roc_auc_score(yw, w["prob"]) if 0 < yw.sum() < len(yw) else np.nan,
                            "avg_R_all": w["ret_R"].mean(), "avg_R_picks": w.loc[w["prob"] >= thr, "ret_R"].mean()})

    preds, summary, windows = pd.concat(preds, ignore_index=True), pd.DataFrame(summary), pd.DataFrame(windows)
    tag = f"g{args.group}" + ("_compare" if args.compare else "") + ("_" + "_".join(sets).replace("+", "") if sets else "")
    preds.to_parquet(PROCESSED_DIR / f"wf_preds_{tag}.parquet", index=False)
    summary.to_csv(PROCESSED_DIR / f"wf_summary_{tag}.csv", index=False)
    windows.to_csv(PROCESSED_DIR / f"wf_windows_{tag}.csv", index=False)

    print("\nsummary:")
    print(summary[["features", "n", "roc_auc", "roc_lo", "roc_hi", "pr_auc", "base_rate", "precision",
                   "avg_R_all", "avg_R_picks", "picks_trades", "picks_max_drawdown_R"]].round(3).to_string(index=False))
    print("\nroc_auc per window:")
    print(windows.pivot(index="window", columns="features", values="roc_auc").round(3).to_string())

    # charts
    out = DATA_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, w in windows.groupby("features", sort=False):
        ax.plot(w["start"], w["roc_auc"], marker="o", label=name)
    ax.axhline(0.5, color="gray", linestyle="--", label="guessing")
    ax.set(xlabel="Window start", ylabel="ROC-AUC", title="Walk forward ROC-AUC per window", ylim=(0.2, 0.8))
    ax.legend()
    save(fig, out, f"wf_roc_by_window_{tag}.png")

    last = preds[preds["features"] == summary["features"].iloc[-1]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, t in [("every setup", last), ("model picks", last[last["prob"] >= thr])]:
        eq = equity_curve(t)
        ax.plot(eq.index, eq.to_numpy(), label=f"{name} ({len(t)} trades)")
    ax.set(xlabel="Exit date", ylabel="Cumulative R", title=f"Walk forward equity curve ({summary['features'].iloc[-1]})")
    ax.legend()
    save(fig, out, f"wf_equity_{tag}.png")

    with open(PROCESSED_DIR / f"wf_settings_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"walk_forward": wf, "threshold": thr, "sets": feature_sets(mcfg["features"], args.compare, sets)}, f, indent=2)


if __name__ == "__main__":
    main()
