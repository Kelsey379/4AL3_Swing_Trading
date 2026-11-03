import numpy as np
import pandas as pd

# time based splits w/ purging
def _purge(df, rows, start, embargo_days): # keep only rows w/ trade closed before start
    cutoff = start-pd.Timedelta(days=embargo_days)
    return rows[(df.loc[rows, "exit_time"] < cutoff).to_numpy()]


def time_split(df, test_frac=0.2, embargo_days=3): # df sorted by signal_time will output train and test row pos
    n =len(df)
    if n < 10:
        raise ValueError("need more setups to split")
    test_start = df["signal_time"].iloc[int(n * (1 - test_frac))]

    test_idx = np.flatnonzero((df["signal_time"] >= test_start).to_numpy())
    before = np.flatnonzero((df["signal_time"] < test_start).to_numpy())
    train_idx = _purge(df, before, test_start, embargo_days)
    return train_idx, test_idx


def walk_forward_folds(df, n_folds=4, embargo_days=3): # expanding window in time order
    chunks = np.array_split(np.arange(len(df)), n_folds + 1)
    folds= []
    for k in range(1, n_folds + 1):
        val_idx = chunks[k]
        val_start =df["signal_time"].iloc[val_idx[0]]
        train_idx = _purge(df, np.concatenate(chunks[:k]), val_start, embargo_days)
        folds.append((train_idx, val_idx))
    return folds
