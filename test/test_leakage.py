import pandas as pd
import pytest
from common.leakage import check_feature_times, check_truncation

TZ = "America/New_York"
SETUPS = pd.DataFrame({"setup_id": ["a", "b"],
                       "signal_bar_end": pd.to_datetime(["2026-10-01 13:30", "2026-10-05 16:00"]).tz_localize(TZ)})


def test_feature_times_ok():
    feats = pd.DataFrame({"setup_id": ["a", "b"], "asof": pd.to_datetime(["2026-09-30 16:00", "2026-10-05 15:59"]).tz_localize(TZ)})
    assert check_feature_times(SETUPS, feats, "asof") == 2


def test_feature_at_bar_close_is_leakage(): # at the close counts too, you cant trade on it
    feats=pd.DataFrame({"setup_id": ["a"], "asof": pd.to_datetime(["2026-10-01 13:30"]).tz_localize(TZ)})
    with pytest.raises(ValueError):
        check_feature_times(SETUPS, feats, "asof")


def _avg_before(data, setups): # honest builder, mean of values before each setup
    rows = [{"setup_id": s, "x": data.loc[data["t"] < end, "v"].mean()} for s, end in zip(setups["setup_id"], setups["signal_bar_end"])]
    return pd.DataFrame(rows)


def _avg_all(data, setups): # leaky builder, uses every value incl future
    return pd.DataFrame({"setup_id": setups["setup_id"], "x": data["v"].mean()})


DATA = pd.DataFrame({"t": pd.date_range("2026-09-25", periods=15, freq="D", tz=TZ), "v": range(15)})


def test_truncation_passes_honest_builder():
    assert check_truncation(_avg_before, DATA, SETUPS, [pd.Timestamp("2026-10-02", tz=TZ)], "t")


def test_truncation_catches_leaky_builder():
    
    with pytest.raises(AssertionError):
        check_truncation(_avg_all, DATA, SETUPS, [pd.Timestamp("2026-10-02", tz=TZ)], "t")
