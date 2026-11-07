import numpy as np
import pandas as pd

from common.config import load_strategy

# relative features, compare each stock to spy + to every other ticker at the same bar
# macro is the same for every setup at one time so it cant pick b/t them, these can

REL_FEATS = ["rs_spy_20", "rs_spy_short", "rs_rank_20"]
BENCH = "SPY"


def relative_frame(bars, cfg=None, n=20): # bars = 4h bars w/ ticker col (all tickers + SPY), bar t only uses bars <= t
    cfg = cfg or load_strategy()
    k = cfg["detector"]["pullback_lookback_bars"]
    close = bars.reset_index().pivot_table(index="bar_start", columns="ticker", values="Close").sort_index()
    if BENCH not in close.columns:
        raise ValueError(f"need {BENCH} bars, run python -m scripts.pull_ohlcv --benchmarks")

    ret_n = close / close.shift(n) - 1
    ret_k = close / close.shift(k) - 1
    stocks = [c for c in close.columns if c not in ("SPY", "QQQ")]
    out = {
        "rs_spy_20": ret_n[stocks].sub(ret_n[BENCH], axis=0),  # beat the market over ~2 weeks?
        "rs_spy_short": ret_k[stocks].sub(ret_k[BENCH], axis=0),  # did it dip w/ the market or on its own
        "rs_rank_20": ret_n[stocks].rank(axis=1, pct=True),  # 1 = strongest ticker at that bar
    }
    long = pd.concat({f: v.stack(future_stack=True) for f, v in out.items()}, axis=1)
    long.index.names = ["bar_start", "ticker"]
    return long


def build_relative_features(bars, setups, cfg=None, n=20): # one row per setup, rel_data_time = signal bar close
    f = relative_frame(bars, cfg, n)
    key = pd.MultiIndex.from_arrays([pd.DatetimeIndex(setups["signal_time"]), setups["ticker"]])
    vals = f.reindex(key)  # missing bar = NaN row
    ends = bars.reset_index().drop_duplicates("bar_start").set_index("bar_start")["bar_end"]
    out = pd.DataFrame({"setup_id": setups["setup_id"].to_numpy(),
                        **{c: vals[c].to_numpy(dtype=float) for c in REL_FEATS}})
    out["rel_data_time"] = ends.reindex(pd.DatetimeIndex(setups["signal_time"])).to_numpy()
    return out
