import numpy as np
import pandas as pd
import pytest
from classifiers.xgb import walk_forward_predict

MCFG = {"split": {"embargo_days": 3}, "xgb": {"n_estimators": 30},
        "grid": {"max_depth": [1, 2], "learning_rate": [0.1]}}


def setups(n=400, seed=0): # one setup a day, x > 0.8 = win (~20%)
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01 09:30", periods=n, freq="D", tz="America/New_York")
    x = rng.normal(size=n)
    return pd.DataFrame({"setup_id": [f"s{i}" for i in range(n)], "signal_time": t,
                         "exit_time": t + pd.Timedelta(days=4), "x": x, "noise": rng.normal(size=n),
                         "label": (x > 0.8).astype(int)})


def test_each_setup_predicted_once_and_learns():
    df = setups()
    p = walk_forward_predict(df, ["x", "noise"], MCFG, n_windows=4, inner_folds=2, min_train=50)
    assert p["setup_id"].is_unique
    assert len(p) == 400 - 80  # first chunk is only ever train
    y = df.set_index("setup_id").loc[p["setup_id"], "label"].to_numpy()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p["prob"]) > 0.95


def test_no_peeking(): # every training trade closed before its window started (purge + embargo)
    df = setups()
    p = walk_forward_predict(df, ["x"], MCFG, n_windows=4, inner_folds=2, min_train=50)
    start = p.merge(df[["setup_id", "signal_time"]], on="setup_id").groupby("window")["signal_time"].min()
    last = p.groupby("window")["train_last_exit"].first()
    assert (last < start - pd.Timedelta(days=3)).all()


def test_min_train_skips_early_windows():
    df = setups()
    p = walk_forward_predict(df, ["x"], MCFG, n_windows=4, inner_folds=2, min_train=150)
    assert p["window"].min() > 1 and p["n_train"].min() >= 150


def test_min_train_too_big():
    with pytest.raises(ValueError):
        walk_forward_predict(setups(), ["x"], MCFG, n_windows=4, inner_folds=2, min_train=10_000)
