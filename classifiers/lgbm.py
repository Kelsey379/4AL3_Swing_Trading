import itertools
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from classifiers.splits import walk_forward_folds
from classifiers.xgb import SEED, evaluate


#similar to classifiers.xgb.training: same class-imbalance weighting, one thread + fixed seed for repeatable runs.
# eval_set = [(X, y), ...] with matching eval_names, to track pr_auc ("average_precision") per tree
def training(X_train, y_train, params, eval_set=None, eval_names=None):
    pos = max(int(np.sum(y_train == 1)), 1)
    neg = int(np.sum(y_train == 0))
    model = LGBMClassifier(objective="binary", scale_pos_weight=neg / pos, random_state=SEED, n_jobs=1,
                           verbose=-1, **params)
    model.fit(X_train, y_train, eval_set=eval_set, eval_names=eval_names, eval_metric="average_precision")
    return model


def cross_validation(X, y, folds, params): # walk forward folds, basically same as xgb.cross_validation
    scores = []
    for train_idx, val_idx in folds:
        model = training(X[train_idx], y[train_idx], params)
        prob = model.predict_proba(X[val_idx])[:, 1]
        scores.append(evaluate(y[val_idx], prob)["pr_auc"])
    return scores


def tune_model(X, y, folds, base_params, grid): # try every combo, score = mean pr_auc over folds
    records = []
    keys = list(grid)
    for combo in itertools.product(*grid.values()):
        params = {**base_params, **dict(zip(keys, combo))}
        scores = cross_validation(X, y, folds, params)
        records.append({**dict(zip(keys, combo)), "mean_pr_auc": np.nanmean(scores), "std_pr_auc": np.nanstd(scores),
                        **{f"fold_{k + 1}": s for k, s in enumerate(scores)}})

    results = pd.DataFrame(records).sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)
    best = {**base_params, **{k: results.loc[0, k] for k in keys}}
    best = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in best.items()}
    return best, results


# share of total split gain per feature (sums to 1)
def gain_share(model, feats):
    gain = model.booster_.feature_importance(importance_type="gain").astype(float)
    total = gain.sum()
    return pd.Series(gain / total if total > 0 else gain, index=feats)

#same as in xgb.py, but for lgbm
def walk_forward_predict(df, feats, mcfg, n_windows=8, inner_folds=3, min_train=250):
    emb = mcfg["split"]["embargo_days"]
    X, y = df[feats].to_numpy(dtype=float), df["label"].to_numpy()
    parts = []
    for k, (tr, te) in enumerate(walk_forward_folds(df, n_windows, emb), 1):
        if len(tr) < min_train:  # too little history to learn anything yet
            continue
        sub = df.iloc[tr].reset_index(drop=True)
        best, _ = tune_model(X[tr], y[tr], walk_forward_folds(sub, inner_folds, emb), mcfg["lgbm"], mcfg["lgbm_grid"])
        model = training(X[tr], y[tr], best)
        parts.append(pd.DataFrame({"setup_id": df["setup_id"].iloc[te].to_numpy(), "window": k,
                                   "prob": model.predict_proba(X[te])[:, 1],
                                   "n_train": len(tr), "train_last_exit": df["exit_time"].iloc[tr].max()}))
    if not parts:
        raise ValueError("no window had enough training setups, lower min_train")
    return pd.concat(parts, ignore_index=True)
