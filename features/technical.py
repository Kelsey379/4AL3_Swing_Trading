import numpy as np
import pandas as pd

from common.config import load_strategy
from features.indicators import macd

# extra technical features for the model (macd, volatility, volume), detector doesnt use any of these
# all scaled so diff priced tickers compare, same idea as add_basic_features

TECH_FEATS = ["macd_pct", "macd_hist_pct", "macd_hist_chg", "vol_long", "vol_ratio",
              "rel_volume", "pullback_volume"]


def _session_rel_volume(bars, n): # vol vs avg of last n bars from SAME session (morning bar is 4h, afternoon 2.5h)
    session = bars.index.hour >= 13
    rel = pd.Series(np.nan, index=bars.index)
    for s in (False, True):
        v = bars.loc[session == s, "Volume"].astype(float)
        avg = v.rolling(n, min_periods=n).mean().shift(1)  # shift so bar isnt in its own avg
        rel.loc[v.index] = v / avg.replace(0, np.nan)
    return rel


def technical_frame(bars, cfg=None): # one row per bar, bar t only uses bars <= t
    cfg = cfg or load_strategy()
    t = cfg["technical"]
    k = cfg["detector"]["pullback_lookback_bars"]
    c = bars["Close"].astype(float)

    line, _, hist = macd(c, t["macd_fast"], t["macd_slow"], t["macd_signal"])
    logret = np.log(c).diff()
    vol_s = logret.rolling(t["vol_short"], min_periods=t["vol_short"]).std()
    vol_l = logret.rolling(t["vol_long"], min_periods=t["vol_long"]).std()
    rel_vol = _session_rel_volume(bars, t["volume_lookback"])

    return pd.DataFrame({
        "macd_pct": line / c,
        "macd_hist_pct": hist / c,
        "macd_hist_chg": (hist - hist.shift(k)) / c,  # >0 = momentum turning back up thru the pullback
        "vol_long": vol_l,  # per bar, not annualized
        "vol_ratio": vol_s / vol_l.replace(0, np.nan),  # >1 = vol expanding lately
        "rel_volume": rel_vol,  # signal bar volume vs normal
        "pullback_volume": rel_vol.shift(1).rolling(k, min_periods=k).mean(),  # bars before signal. low = quiet dip (good?)
        "tech_data_time": bars["bar_end"],
    }, index=bars.index)


def build_technical_features(data, setups, cfg=None): # data = 4h bars w/ ticker col (can be many tickers)
    # tech_data_time = signal bar close, same bar the detector already uses for close/rsi. entry is next open so its ok
    cfg = cfg or load_strategy()
    parts = []
    for tk, s in setups.groupby("ticker", sort=False):
        bars = data[data["ticker"] == tk]
        if bars.empty:
            f = pd.DataFrame(index=pd.Index(s["signal_time"]), columns=TECH_FEATS + ["tech_data_time"])
        else:
            f = technical_frame(bars, cfg).reindex(pd.DatetimeIndex(s["signal_time"]))  # missing bar = NaN row
        parts.append(pd.DataFrame({"setup_id": s["setup_id"].to_numpy(),
                                   **{col: f[col].to_numpy() for col in TECH_FEATS + ["tech_data_time"]}}))
    cols = ["setup_id"] + TECH_FEATS + ["tech_data_time"]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out[TECH_FEATS] = out[TECH_FEATS].astype(float)
    out["tech_data_time"] = pd.to_datetime(out["tech_data_time"])
    # back in setups order
    return out.set_index("setup_id").reindex(setups["setup_id"]).reset_index()[cols]
