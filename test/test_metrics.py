import numpy as np
import pandas as pd
import pytest
from backtest.metrics import daily_returns, equity_curve, max_drawdown, sharpe, summarize, win_rate


def trades(rets, start="2026-09-01 16:00"): # one closing per business day
    t = pd.bdate_range(start, periods=len(rets), tz="America/New_York")
    return pd.DataFrame({"exit_time": t, "ret_R": rets})


def test_win_rate():
    assert np.isclose(win_rate(trades([2.0, -1.0, 0.5, -1.0])), 0.5)


def test_win_rate_counts_profitable_time_exit():
    assert np.isclose(win_rate(trades([0.3, -1.0])), 0.5)  # would be 0 but made $


def test_win_rate_needs_trades():
    with pytest.raises(ValueError):
        win_rate(trades([]))


def test_pending_trades_ignored():
    t= trades([2.0, -1.0, np.nan])
    assert np.isclose(win_rate(t), 0.5)
    assert summarize(t)["trades"] == 2


def test_equity_curve_is_running_sum():
    assert equity_curve(trades([2.0, -1.0, 2.0])).tolist() == [2.0, 1.0, 3.0]


def test_max_drawdown():
    # peak at 3 drop to 0 = 3R drawdown
    assert np.isclose(max_drawdown(trades([2.0, 1.0, -1.0, -1.0, -1.0, 2.0])), 3.0)


def test_max_drawdown_first_trade_loss():
    assert np.isclose(max_drawdown(trades([-1.0, 2.0])), 1.0)


def test_no_drawdown_if_only_wins():
    assert max_drawdown(trades([1.0, 2.0, 0.5])) == 0


def test_daily_returns_fills_gaps_and_sums_same_day():
    t = pd.DataFrame({"exit_time": pd.to_datetime(["2026-09-01 13:30", "2026-09-01 16:00", "2026-09-03 16:00"])
                      .tz_localize("America/New_York"), "ret_R": [1.0, -1.0, 2.0]})
    d =daily_returns(t)
    assert d.tolist() == [0.0, 0.0, 2.0]  # sep 1 nets to 0, sep 2 had no exits


def test_sharpe_by_hand():
    rets = [1.0, -1.0, 2.0, 0.0]
    expected = np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252)
    assert np.isclose(sharpe(trades(rets)), expected)


def test_sharpe_undefined_with_no_variation():
    assert np.isnan(sharpe(trades([1.0, 1.0, 1.0])))
