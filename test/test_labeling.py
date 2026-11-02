import numpy as np
import pandas as pd
from helpers import bars_from, hourly_frame
from labeling.labeler import drop_overlapping, label_setups

# atr = 1 so stop = entry - 1.5 and target = entry + 3. entry = open of bar 1 = 100
CFG = {"labels": {"stop_atr": 1.5, "target_atr": 3.0, "horizon_bars": 4, "one_position_per_ticker": True}}

SIGNAL = (99, 100, 98, 100)  # bar 0, the signal bar
FLAT = (100, 100.5, 99.5, 100)


def run(rows, hourly=None): # label one setup on bar 0
    bars = bars_from(rows)
    setup = pd.DataFrame({"setup_id": ["X_0"], "ticker": ["X"], "signal_time": [bars.index[0]], "atr": [1.0]})
    return label_setups(setup, bars, CFG, hourly=hourly).iloc[0]


def test_target_first_is_win():
    r = run([SIGNAL, (100, 101, 99.5, 100.5), (100.5, 103.5, 100, 103)])
    assert r["barrier"] == "tp" and r["label"] == 1
    assert r["exit_price"] == 103 and r["bars_held"] == 2
    assert np.isclose(r["ret_R"], 2.0)


def test_stop_first_is_loss():
    r = run([SIGNAL, (100, 100.5, 98.4, 99), FLAT])
    assert r["barrier"] == "sl" and r["label"] == 0
    assert np.isclose(r["ret_R"], -1.0)


def test_time_limit_is_loss():
    r = run([SIGNAL, FLAT, FLAT, FLAT, (100, 100.5, 99.5, 100.3), FLAT])
    assert r["barrier"] == "time" and r["label"] == 0
    assert r["exit_price"] == 100.3 and r["bars_held"] == 4
    assert r["ret_R"] > 0 # up at the time limit but still a loss


def test_entry_is_next_open():
    r = run([(99, 100, 98, 99.7), (100, 100.2, 99.9, 100), FLAT, FLAT, FLAT])
    assert r["entry_price"] == 100
    assert r["entry_time"] > r["signal_time"]


def test_gap_past_stop_fills_at_open():
    r = run([SIGNAL, FLAT, (97, 97.5, 96, 97)])
    assert r["barrier"] == "sl" and r["exit_price"] == 97
    assert r["ret_R"] < -1  # worse than 1R


def test_same_bar_no_hourly_is_loss():
    r = run([SIGNAL, (100, 104, 98, 100)])
    assert r["barrier"] == "sl" and r["same_bar"]


def test_same_bar_hourly_breaks_tie():
    bars = bars_from([SIGNAL, (100, 104, 98, 100)])
    hourly = hourly_frame(1)
    am = hourly.index[hourly.index < bars["bar_end"].iloc[0]]
    pm = hourly.index[hourly.index >= bars.index[1]]
    hourly.loc[am, ["High", "Low"]] = [100.5, 99.5]
    hourly.loc[pm, ["High", "Low"]] = [[104, 99.5], [100.5, 98], [100.5, 99.5]]  # target hour first

    setup = pd.DataFrame({"setup_id": ["X_0"], "ticker": ["X"], "signal_time": [bars.index[0]], "atr": [1.0]})
    r = label_setups(setup, bars, CFG, hourly=hourly).iloc[0]
    assert r["barrier"] == "tp" and r["same_bar"]


def test_not_enough_bars_is_pending():
    r = run([SIGNAL, FLAT, FLAT])
    assert pd.isna(r["label"]) and pd.isna(r["barrier"])


def test_drop_overlapping():
    t = pd.date_range("2026-09-01 09:30", periods=5, freq="D", tz="America/New_York")
    df = pd.DataFrame({"setup_id": list("abc"), "ticker": "X",
        "signal_time": [t[0], t[1], t[3]], "exit_time": [t[2], t[4], t[4]], "label": [1.0, 0.0, 1.0]})
    assert drop_overlapping(df)["setup_id"].tolist() == ["a", "c"]  # b starts while a is open


def test_open_trade_blocks_later_setups(): # pending trade w/ no exit yet blocks everything after
    t = pd.date_range("2026-09-01 09:30", periods=3, freq="D", tz="America/New_York")
    df = pd.DataFrame({"setup_id": list("ab"), "ticker": "X", "signal_time": [t[0], t[1]],
                       "exit_time": [pd.NaT, pd.NaT], "label": [np.nan, np.nan]})
    assert drop_overlapping(df)["setup_id"].tolist() == ["a"]
