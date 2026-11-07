import numpy as np
import pandas as pd
import pytest
from dashboard.probs import at_threshold, in_group, score_setups


def preds(probs, labels, rets):
    return pd.DataFrame({"ticker": ["A"] * len(probs), "prob": probs, "label": labels, "ret_R": rets})


def test_at_threshold_picks_and_rates():
    s = at_threshold(preds([0.2, 0.6, 0.7, 0.4], [0, 1, 0, 1], [-1.0, 2.0, -1.0, 2.0]), 0.5)
    assert s["setups"] == 4 and s["picked"] == 2
    assert np.isclose(s["win_rate_all"], 0.5) and np.isclose(s["win_rate_picked"], 0.5)
    assert np.isclose(s["avg_r_picked"], 0.5)


def test_at_threshold_nothing_picked():
    s = at_threshold(preds([0.1, 0.2], [0, 1], [-1.0, 2.0]), 0.9)
    assert s["picked"] == 0 and np.isnan(s["win_rate_picked"]) and np.isnan(s["avg_r_picked"])


def test_in_group_all_keeps_everything():
    df = preds([0.1], [0], [-1.0])
    assert len(in_group(df, "all")) == 1


class FakeModel: # prob = rsi / 100 so its easy to check
    def predict_proba(self, X):
        return np.c_[1 - X[:, 0] / 100, X[:, 0] / 100]


def setups():
    return pd.DataFrame({"rsi": [40.0, 50.0], "close": [110.0, 50.0], "ema_fast": [100.0, 48.0], "sma_trend": [90.0, 45.0], "atr": [2.0, 1.0], "signal_time": pd.to_datetime(["2026-10-01 09:30", "2026-10-01 13:30"]).tz_localize("America/New_York")})


def test_score_setups_uses_training_features():
    d = score_setups(setups(), FakeModel(), ["rsi", "atr_pct", "afternoon"])
    assert np.allclose(d["prob"], [0.4, 0.5])


def test_score_setups_missing_feature():
    with pytest.raises(ValueError):
        score_setups(setups(), FakeModel(), ["rsi", "vix_close"])
