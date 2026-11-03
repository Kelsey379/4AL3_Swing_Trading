import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

SEED = 42


def tss(y_true, y_pred):
    y_true = np.asarray(y_true)
    if 0 not in y_true or 1 not in y_true:
        raise ValueError("Both classes must be present in y_true to calculate TSS.")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    return float(tp / (tp + fn) - fp / (fp + tn))


def evaluate(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob)>= threshold).astype(int)
    both = 0 in y_true and 1 in y_true  
    return {
        "n": len(y_true),
        "base_rate": float(np.mean(y_true)),  # precision must be better than this
        "predicted_wins": int(pred.sum()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)) if both else np.nan,
        "roc_auc": float(roc_auc_score(y_true, prob)) if both else np.nan,
        "tss": tss(y_true, pred) if both else np.nan,
    }


def training(X_train, y_train, params, eval_set=None): # eval_set = [(X, y), ...] to track pr_auc per tree
    pos = max(int(np.sum(y_train==1)), 1)
    neg = int(np.sum(y_train == 0))
    model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", scale_pos_weight=neg / pos,
        random_state=SEED, n_jobs=1, **params)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def cross_validation(X, y, folds, params): # walk forward folds
    scores = []

    for train_idx, val_idx in folds:
        model = training(X[train_idx], y[train_idx], params)
        prob = model.predict_proba(X[val_idx])[:, 1]
        scores.append(evaluate(y[val_idx], prob)["pr_auc"])
    return scores


def tune_model(X, y, folds, base_params, grid): # try every combo, score = mean pr_auc over folds
    records= []
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
