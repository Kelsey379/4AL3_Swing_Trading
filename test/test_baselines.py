import numpy as np
import pandas as pd
from backtest.baselines import buy_and_hold, equal_weight_hold, max_open_positions, unfiltered_detector

TZ = "America/New_York"


def closes(vals, start="2026-09-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


def trades(rets, start="2026-09-01"): # one trade per day, entry 9:30 exit 16:00 same day
    d = pd.bdate_range(start, periods=len(rets), tz=TZ)
    return pd.DataFrame({"signal_time": d + pd.Timedelta(hours=9), "entry_time": d + pd.Timedelta(hours=9, minutes=30),
                         "exit_time": d + pd.Timedelta(hours=16), "entry_price": 100.0,
                         "exit_price": [100 + r for r in rets], "ret_R": rets})


def test_buy_and_hold_total_return():
    r = buy_and_hold(closes([100, 110, 99, 120]))
    assert np.isclose(r["total_return"], 0.2)
    assert np.isclose(r["max_dd"], 0.1)  # 110 -> 99


def test_buy_and_hold_window():
    c = closes([100, 50, 60, 66])
    assert np.isclose(buy_and_hold(c, start=c.index[2])["total_return"], 0.1)


def test_equal_weight_averages_daily():
    a, b = closes([100, 110, 110]), closes([100, 90, 90])
    assert np.isclose(equal_weight_hold({"a": a, "b": b})["total_return"], 0.0)  # +10% and -10% = 0 day 1


def test_detector_risk_scaling():
    r = unfiltered_detector(trades([2.0, -1.0, 1.0]), risk=0.01)
    assert r["R_trades"] == 3 and np.isclose(r["R_total_R"], 2.0)
    assert np.isclose(r["total_return"], 1.02 * 0.99 * 1.01 - 1, atol=1e-4)  # compounds, rounded to 4dp


def test_detector_end_day_included(): # end = midnight but setups later that day still count
    t = trades([1.0, 1.0, 1.0])
    end = pd.Timestamp(t["signal_time"].iloc[-1]).tz_localize(None).normalize()
    assert unfiltered_detector(t, end=end)["R_trades"] == 3


def test_max_open_positions():
    t = pd.DataFrame({"entry_time": pd.to_datetime(["2026-09-01 10:00", "2026-09-01 11:00", "2026-09-01 12:00"]),
                      "exit_time": pd.to_datetime(["2026-09-01 12:00", "2026-09-01 13:00", "2026-09-01 14:00"])})
    assert max_open_positions(t) == 2  # first closes as third opens
