import numpy as np
import pandas as pd
from classifiers.dataset import add_basic_features
from classifiers.xgb import evaluate, training, tss, tune_model

PARAMS = {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1}


def test_tss_checks(): 
    assert np.isclose(tss([0, 0, 1, 1], [0, 0, 1, 1]), 1)
    assert np.isclose(tss([0, 0, 1, 1], [0, 0, 0, 0]), 0)
    assert np.isclose(tss([0, 0, 1, 1], [1, 1, 0, 0]), -1)


def test_evaluate_one_class_doesnt_crash():
    r= evaluate([0, 0, 0], [0.1, 0.6, 0.2])
    assert r["precision"] == 0 and np.isnan(r["roc_auc"])


def test_evaluate_numbers():
    r = evaluate([0, 1, 1, 0], [0.2, 0.9, 0.4, 0.7], threshold=0.5)
    assert r["predicted_wins"] == 2
    assert np.isclose(r["precision"], 0.5) and np.isclose(r["recall"], 0.5)


def test_model_learns_an_easy_signal(): # x > 0 = win
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400,2))
    y = (X[:, 0] > 0.8).astype(int)  # ~20% wins 
    model = training(X[:300], y[:300], PARAMS)
    r = evaluate(y[300:], model.predict_proba(X[300:])[:, 1])
    assert r["roc_auc"] > 0.95


def test_tune_model_picks_from_grid():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0.8).astype(int)
    folds = [(np.arange(0, 100), np.arange(100, 200)), (np.arange(0, 200), np.arange(200, 300))]
    best, results = tune_model(X, y, folds, {"n_estimators": 30}, {"max_depth": [1, 2], "learning_rate": [0.1]})
    assert len(results) == 2 and best["max_depth"] in (1, 2)
    assert results["mean_pr_auc"].is_monotonic_decreasing


def test_basic_features():
    s = pd.DataFrame({"close": [110.0], "ema_fast": [100.0], "sma_trend": [80.0], "atr": [2.2],
                      "signal_time": [pd.Timestamp("2026-10-01 13:30", tz="America/New_York")]})
    f = add_basic_features(s).iloc[0]
    assert np.isclose(f["atr_pct"], 0.02) and np.isclose(f["dist_ema"], 0.1)
    assert np.isclose(f["trend_gap"], 0.25) and f["afternoon"] == 1
