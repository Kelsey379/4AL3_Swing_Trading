import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from classifiers.dataset import add_basic_features, add_macro_features
from common.config import DATA_DIR, PROCESSED_DIR, tickers_for

# data side of the dashboard, kept out of app.py so it can be tested


def load_model(group="all"): # model saved by train_xgb
    model = XGBClassifier()
    model.load_model(DATA_DIR / "models" / f"xgb_g{group}.json")
    return model


def score_setups(setups, model, feats): # adds prob col, same features as training
    d = add_basic_features(setups)
    missing = [f for f in feats if f not in d.columns]
    if missing:
        raise ValueError(f"missing features {missing}, model was trained w/ cols these setups dont have")
    d["prob"] = model.predict_proba(d[feats].to_numpy(dtype=float))[:,1]
    return d


def pending_setups(group="all"): # setups w/ no label yet = trade still open or not entered
    s = pd.read_parquet(PROCESSED_DIR / f"setups_g{group}.parquet")
    s =s[s["label"].isna()].copy()
    return add_macro_features(s, group).sort_values("signal_time", ascending=False).reset_index(drop=True)


def test_predictions(group="all"): # held out test set w/ probs from train_xgb
    return pd.read_parquet(PROCESSED_DIR / f"xgb_test_preds_g{group}.parquet")


def in_group(df, group): # filter to one ticker group, all = no filter
    if str(group) == "all":
        return df
    return df[df["ticker"].isin(tickers_for(group))]


def at_threshold(test, threshold): # what you get if you only take setups w/ prob >= threshold
    picked = test[test["prob"] >= threshold]
    n = len(picked)
    return {"setups": len(test), "picked": n,
            "win_rate_all": float(test["label"].mean()) if len(test) else np.nan,
            "win_rate_picked": float(picked["label"].mean()) if n else np.nan,  # = precision
            "avg_r_all": float(test["ret_R"].mean()) if len(test) else np.nan,
            "avg_r_picked": float(picked["ret_R"].mean()) if n else np.nan}
