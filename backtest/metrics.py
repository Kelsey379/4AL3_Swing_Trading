import numpy as np
import pandas as pd
TRADING_DAYS = 252


def _closed(trades): # drop pending trades
    return trades.dropna(subset=["ret_R"]).sort_values("exit_time")


def win_rate(trades): # share of trades that made money
    r =_closed(trades)["ret_R"].astype(float)
    if len(r) == 0:
        raise ValueError("need at least one closed trade for win rate")
    return float(np.mean(r > 0))


def avg_r(trades):
    return float(np.mean(_closed(trades)["ret_R"].astype(float)))


def daily_returns(trades): # R summed by exit day w/ days with no exits =0
    t = _closed(trades)
    day= pd.to_datetime(t["exit_time"]).dt.tz_localize(None).dt.normalize()
    daily = t["ret_R"].astype(float).groupby(day.to_numpy()).sum()
    all_days = pd.bdate_range(daily.index.min(), daily.index.max())
    return daily.reindex(all_days, fill_value=0.0)


def sharpe(trades): # annualized
    d = daily_returns(trades)
    if len(d) < 2 or d.std() == 0:
        return np.nan # undef w/ 1 day/no variation
    return float(d.mean()/d.std() * np.sqrt(TRADING_DAYS))


def equity_curve(trades): #total of R after each trade closes
    t = _closed(trades)
    return pd.Series(t["ret_R"].astype(float).cumsum().to_numpy(), index=t["exit_time"].to_numpy())


def max_drawdown(trades): # biggest drop from a peak in R
    eq = np.r_[0.0,equity_curve(trades).to_numpy()]
    peak = np.maximum.accumulate(eq)
    return float(np.max(peak - eq))



def summarize(trades):
    t = _closed(trades)
    return {
        "trades": len(t),
        "win_rate": round(win_rate(t), 3),
        "avg_R": round(avg_r(t), 3),
        "total_R": round(float(t["ret_R"].sum()), 3),
        "sharpe": round(sharpe(t), 3),
        "max_drawdown_R": round(max_drawdown(t), 3),
    }
