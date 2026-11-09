import numpy as np
import pandas as pd

from backtest.metrics import TRADING_DAYS, daily_returns, summarize

# baselines the model has to beat:
#  buy and hold = just own spy/qqq (or every ticker equal weight) the whole time
#  unfiltered detector = take every setup, no model. if xgb cant beat this the model isnt adding anything
# R isnt comparable to % so detector gets turned into % w fixed risk per trade (risk=0.01 -> 1R = 1% of acct)


def daily_closes(bars): # last close each day from 4h bars, naive dates
    c = bars["Close"].astype(float)
    day = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
    return c.groupby(day).last()


def _stats(rets): # daily % returns -> summary, same sharpe formula as metrics.sharpe
    rets = rets.dropna()
    if len(rets) == 0:
        return {"days": 0, "total_return": np.nan, "cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    eq = (1 + rets).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])
    dd = 1 - np.r_[1.0, eq.to_numpy()] / peak
    years = len(rets) / TRADING_DAYS
    sd = rets.std() if len(rets) > 1 else 0  # sharpe undef w/ 1 day
    return {
        "days": len(rets),
        "total_return": round(float(eq.iloc[-1] - 1), 4),
        "cagr": round(float(eq.iloc[-1] ** (1 / years) - 1), 4),
        "sharpe": round(float(rets.mean() / sd * np.sqrt(TRADING_DAYS)), 3) if sd > 0 else np.nan,
        "max_dd": round(float(dd.max()), 4),  # as a fraction of peak, not R
    }


def _window(s, start, end):
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return s


def buy_and_hold(closes, start=None, end=None): # one ticker, buy first close in window sell last
    return _stats(_window(closes, start, end).pct_change())


def equal_weight_hold(closes_by_ticker, start=None, end=None): # all tickers equal weight, rebalanced daily (simplification)
    rets = pd.DataFrame({t: c.pct_change() for t, c in closes_by_ticker.items()})
    return _stats(_window(rets.mean(axis=1, skipna=True), start, end))


def detector_daily_returns(trades, risk=0.01): # R by exit day * risk = % of account
    return daily_returns(trades) * risk


def unfiltered_detector(trades, start=None, end=None, risk=0.01): # every setup taken, window by signal time
    day = pd.to_datetime(trades["signal_time"]).dt.tz_localize(None).dt.normalize()  # by day so end day is included
    keep = pd.Series(True, index=trades.index)
    if start is not None:
        keep &= day >= start
    if end is not None:
        keep &= day <= end
    t = trades[keep].dropna(subset=["ret_R"])
    if t.empty:
        return {"trades": 0}
    pct = (t["exit_price"].astype(float) / t["entry_price"].astype(float) - 1)
    out = {f"R_{k}": v for k, v in summarize(t).items()}  # in R, same as metrics
    out["avg_trade_pct"] = round(float(pct.mean()), 4)  # raw stock move per trade, no sizing
    out["max_open"] = max_open_positions(t)  # no cap on positions so % numbers assume this many at once
    out.update(_stats(detector_daily_returns(t, risk)))
    return out


def max_open_positions(trades): # most trades open at the same time
    ev = pd.concat([pd.Series(1, index=pd.to_datetime(trades["entry_time"])),
                    pd.Series(-1, index=pd.to_datetime(trades["exit_time"]))]).sort_index(kind="stable")
    ev = ev.groupby(level=0).sum()  # exits + entries at same stamp net out
    return int(ev.cumsum().max()) if len(ev) else 0
