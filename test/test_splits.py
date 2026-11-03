import numpy as np
import pandas as pd
import pytest
from classifiers.splits import time_split, walk_forward_folds


def setups(n=100, hold_days=4): # one setup a day lasts hold_days
    t = pd.date_range("2026-01-01 09:30", periods=n, freq="D", tz="America/New_York")
    return pd.DataFrame({"signal_time": t, "exit_time": t + pd.Timedelta(days=hold_days)})


def test_test_set_is_the_latest_setups():
    df = setups()
    train_idx, test_idx = time_split(df, test_frac=0.2, embargo_days=3)
    assert len(test_idx) == 20
    assert df["signal_time"].iloc[train_idx].max() < df["signal_time"].iloc[test_idx].min()


def test_train_trades_close_before_test_starts(): # purge + embargo
    df = setups()
    train_idx, test_idx = time_split(df, test_frac=0.2, embargo_days=3)
    test_start = df["signal_time"].iloc[test_idx].min()
    assert (df["exit_time"].iloc[train_idx] < test_start - pd.Timedelta(days=3)).all()
    assert len(train_idx) == 80 - 7  # last 7 train trades overlap (4 day hold + 3 day embargo)


def test_no_row_in_both():
    train_idx, test_idx =time_split(setups())
    assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_too_few_setups():
    with pytest.raises(ValueError):
        time_split(setups(5))


def test_walk_forward_folds_expand_and_dont_peek():
    df = setups(100)
    
    folds = walk_forward_folds(df, n_folds=4, embargo_days=3)
    assert len(folds) == 4
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes)  # train grows each fold

    for train_idx, val_idx in folds:
        val_start = df["signal_time"].iloc[val_idx].min()
        assert (df["exit_time"].iloc[train_idx] < val_start - pd.Timedelta(days=3)).all()
