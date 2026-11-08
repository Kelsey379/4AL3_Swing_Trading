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
from classifiers.lgbm import evaluate, walk_forward_predict
from common.config import DATA_DIR, PROCESSED_DIR, load_yaml, tickers_for
from scripts.walk_forward_xgb import feature_sets, roc_ci, save  # shared so both models use the SAME feature sets

# lightgbm version of scripts.walk_forward_xgb: retrain before each time window, ~1000 out of sample preds
# instead of the ~266 in the test set. same windows, settings, feature sets and metrics, so the models compare.
# python -m scripts.walk_forward_lgbm --group all            <- features in model.yaml
# python -m scripts.walk_forward_lgbm --group all --compare  <- every feature set side by side
# python -m scripts.walk_forward_lgbm --group all --sets detector,det+macro   <- just some
# if scripts.walk_forward_xgb was run with the same flags, the two are printed side by side at the end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="1, 2, 3 or all")
    ap.add_argument("--compare", action="store_true", help="run each feature set")
    ap.add_argument("--sets", default="", help="comma list, e.g. detector,det+macro,all")
    args = ap.parse_args()

    mcfg = load_yaml("model.yaml")
    if "lgbm" not in mcfg or "lgbm_grid" not in mcfg:
        raise SystemExit("add the lgbm: and lgbm_grid: blocks to config/model.yaml first")
    wf, thr = mcfg["walk_forward"], mcfg["threshold"]
    sets = [s for s in args.sets.split(",") if s]
    df = load_dataset(args.group)
    missing = [f for f in mcfg["features"] if f not in df.columns]
    if missing:
        raise SystemExit(f"missing features {missing}, run python -m scripts.build_macro_features --group {args.group}")
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
               **({f"picks_{k}": v for k, v in summarize(picks).items()} if len(picks) else {"picks_trades": 0})}
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
    preds.to_parquet(PROCESSED_DIR / f"wf_lgbm_preds_{tag}.parquet", index=False)
    summary.to_csv(PROCESSED_DIR / f"wf_lgbm_summary_{tag}.csv", index=False)
    windows.to_csv(PROCESSED_DIR / f"wf_lgbm_windows_{tag}.csv", index=False)

    print("\nsummary:")
    show = ["features", "n", "roc_auc", "roc_lo", "roc_hi", "pr_auc", "base_rate", "precision", "avg_R_all", "avg_R_picks", "picks_trades"]
    print(summary[[c for c in show + ["picks_max_drawdown_R"] if c in summary.columns]].round(3).to_string(index=False))
    print("\nroc_auc per window:")
    print(windows.pivot(index="window", columns="features", values="roc_auc").round(3).to_string())

    xgb_path = PROCESSED_DIR / f"wf_summary_{tag}.csv"  # xgboost run with the same flags, if it exists
    if xgb_path.exists():
        cols = [c for c in show if c in summary.columns]
        both = pd.concat([pd.read_csv(xgb_path)[cols].assign(model="xgboost"), summary[cols].assign(model="lightgbm")])
        print("\nxgboost vs lightgbm (same windows, same feature sets):")
        print(both.sort_values(["features", "model"]).round(3).to_string(index=False))

    # charts, in their own folder so the xgboost ones are never overwritten
    out = DATA_DIR / "plots" / "lgbm"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, w in windows.groupby("features", sort=False):
        ax.plot(w["start"], w["roc_auc"], marker="o", label=name)
    ax.axhline(0.5, color="gray", linestyle="--", label="guessing")
    ax.set(xlabel="Window start", ylabel="ROC-AUC", title="LightGBM walk forward ROC-AUC per window", ylim=(0.2, 0.8))
    ax.legend()
    save(fig, out, f"wf_roc_by_window_{tag}.png")

    last = preds[preds["features"] == summary["features"].iloc[-1]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, t in [("every setup", last), ("model picks", last[last["prob"] >= thr])]:
        eq = equity_curve(t)
        ax.plot(eq.index, eq.to_numpy(), label=f"{name} ({len(t)} trades)")
    ax.set(xlabel="Exit date", ylabel="Cumulative R", title=f"LightGBM walk forward equity curve ({summary['features'].iloc[-1]})")
    ax.legend()
    save(fig, out, f"wf_equity_{tag}.png")

    with open(PROCESSED_DIR / f"wf_lgbm_settings_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"walk_forward": wf, "threshold": thr, "lgbm": mcfg["lgbm"], "lgbm_grid": mcfg["lgbm_grid"],
                   "sets": feature_sets(mcfg["features"], args.compare, sets)}, f, indent=2)


if __name__ == "__main__":
    main()
