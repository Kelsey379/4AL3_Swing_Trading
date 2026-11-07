import numpy as np
import pandas as pd
import pytest
from common.config import load_strategy
from common.leakage import check_truncation
from features.indicators import ema, macd
from features.technical import TECH_FEATS, build_technical_features, technical_frame
from helpers import random_walk_bars
from labeling.detector import detect_ma_pullbacks


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_lookahead(seed): # cutting future bars cant change any older row
    cfg = load_strategy()
    bars = random_walk_bars(seed=seed)
    full = technical_frame(bars, cfg)
    for cut in (100, 150, 200):
        pd.testing.assert_frame_equal(technical_frame(bars.iloc[:cut], cfg), full.iloc[:cut])


def test_truncation_w_leakage_helper(): # same check the script runs
    cfg = load_strategy()
    bars = random_walk_bars(seed=2).assign(ticker="X")
    setups = detect_ma_pullbacks(bars, cfg, "X")
    assert len(setups) > 0
    cuts = [e + pd.Timedelta(minutes=1) for e in setups["signal_bar_end"].iloc[::3]]
    assert check_truncation(lambda d, s: build_technical_features(d, s, cfg), bars, setups, cuts, "bar_end")


def test_macd_is_ema_gap():
    c = pd.Series(np.linspace(100, 130, 80))
    line, sig, hist = macd(c, 12, 26, 9)
    assert np.allclose((ema(c, 12) - ema(c, 26)).dropna(), line.dropna())
    assert np.allclose((line - sig).dropna(), hist.dropna())
    assert line.iloc[-1] > 0  # uptrend = fast above slow


def test_rel_volume_per_session(): # afternoon always half of morning shouldnt look like low volume
    cfg = load_strategy()
    bars = random_walk_bars(seed=1)
    bars["Volume"] = np.where(bars.index.hour >= 13, 500.0, 1000.0)
    rel = technical_frame(bars, cfg)["rel_volume"].dropna()
    assert len(rel) > 0 and np.allclose(rel, 1.0)


def test_setup_order_kept_and_missing_ticker_nan():
    cfg = load_strategy()
    bars = random_walk_bars(seed=2).assign(ticker="X")
    s = detect_ma_pullbacks(bars, cfg, "X")
    other = s.head(2).assign(ticker="Y", setup_id=["Y_1", "Y_2"])  # no bars for Y
    both = pd.concat([other, s], ignore_index=True)
    out = build_technical_features(bars, both, cfg)
    assert out["setup_id"].tolist() == both["setup_id"].tolist()
    assert out.loc[out["setup_id"].str.startswith("Y"), TECH_FEATS].isna().all().all()
    assert out.loc[~out["setup_id"].str.startswith("Y"), "macd_pct"].notna().all()
    assert (out["tech_data_time"].iloc[2:].to_numpy() == s["signal_bar_end"].to_numpy()).all()
