import pandas as pd
import pytest
from common.config import load_strategy
from helpers import random_walk_bars
from labeling.detector import detect_ma_pullbacks
from labeling.labeler import label_setups


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_lookahead(seed): # cutting off future bars cant change any earlier setup
    cfg = load_strategy()
    bars = random_walk_bars(seed=seed)
    full = detect_ma_pullbacks(bars, cfg, "X")
    assert len(full) > 0, "test data should have some setups"

    for cut in (130, 170, 210):
        part = detect_ma_pullbacks(bars.iloc[:cut], cfg, "X")
        expected = full[full["signal_time"] < bars.index[cut]].reset_index(drop=True)
        pd.testing.assert_frame_equal(part, expected)


def test_downtrend_has_no_setups(): # needs price above sma50
    cfg = load_strategy()
    bars = random_walk_bars(seed=1, drift=-0.002)
    assert len(detect_ma_pullbacks(bars, cfg, "X")) == 0


def test_setups_are_in_an_uptrend():
    cfg = load_strategy()
    setups = detect_ma_pullbacks(random_walk_bars(seed=2), cfg, "X")
    assert (setups["close"] > setups["sma_trend"]).all()
    assert (setups["ema_fast"] > setups["sma_trend"]).all()
    assert (setups["close"] > setups["ema_fast"]).all()  # bounced back above


def test_output_has_ticker_and_time():
    cfg = load_strategy()
    setups = detect_ma_pullbacks(random_walk_bars(seed=3), cfg, "ABC")
    assert (setups["ticker"] == "ABC").all()
    assert setups["signal_time"].notna().all()
    assert setups["setup_id"].is_unique


def test_entry_always_after_signal():
    cfg = load_strategy()
    bars = random_walk_bars()
    labeled = label_setups(detect_ma_pullbacks(bars, cfg, "X"), bars, cfg)
    ok = labeled.dropna(subset=["entry_time"])

    assert (ok["entry_time"] > ok["signal_time"]).all()
    assert (ok["entry_time"] >= ok["signal_bar_end"]).all()
