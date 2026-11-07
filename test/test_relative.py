import numpy as np
import pandas as pd
import pytest
from common.leakage import check_truncation
from features.relative import REL_FEATS, build_relative_features
from helpers import hourly_frame
from data_collection.ohlcv import resample_4h

def bars(trends): # ticker -> % move per hourly bar
    parts = []
    for t, g in trends.items():
        h = hourly_frame(30, closes=100 * (1 + g) ** np.arange(30 * 7))
        parts.append(resample_4h(h).assign(ticker=t))
    return pd.concat(parts)


def setups(ticker="A", bar=25):
    b = bars({"SPY": 0.0})
    t = b.index[bar]
    return pd.DataFrame({"setup_id": ["x"], "ticker": [ticker], "signal_time": [t],
                         "signal_bar_end": [b["bar_end"].iloc[bar]]})


def test_stronger_than_spy_is_positive_and_ranked_top():
    data = bars({"SPY": 0.001, "A": 0.003, "B": 0.0, "C": -0.002})
    f = build_relative_features(data, setups("A")).iloc[0]
    assert f["rs_spy_20"] > 0 and f["rs_spy_short"] > 0
    assert f["rs_rank_20"] == 1.0  # strongest of A, B, C


def test_weakest_ranked_bottom():
    data = bars({"SPY": 0.001, "A": 0.003, "B": 0.0, "C": -0.002})
    f = build_relative_features(data, setups("C")).iloc[0]
    assert f["rs_spy_20"] < 0 and np.isclose(f["rs_rank_20"], 1 / 3)


def test_data_time_is_signal_bar_close():
    s = setups("A")
    f = build_relative_features(bars({"SPY": 0.001, "A": 0.002}), s)
    assert f["rel_data_time"].iloc[0] == s["signal_bar_end"].iloc[0]


def test_future_bars_dont_change_it():
    data = bars({"SPY": 0.001, "A": 0.003, "B": 0.0})
    s = setups("A")
    assert check_truncation(build_relative_features, data, s, [s["signal_bar_end"].iloc[0] + pd.Timedelta(minutes=1)], "bar_end")


def test_needs_spy():
    with pytest.raises(ValueError):
        build_relative_features(bars({"A": 0.001}), setups("A"))


def test_feats_listed():
    assert set(REL_FEATS) <= set(build_relative_features(bars({"SPY": 0.001, "A": 0.002}), setups("A")).columns)
