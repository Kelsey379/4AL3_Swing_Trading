import numpy as np
import pandas as pd

from data_collection.gdelt import SUM_COLS

TZ = "America/New_York"
UTC_NS = "datetime64[ns, UTC]"

VIX_FEATS = ["vix_close", "vix_chg_5d", "vix_vs_ma20"]
GDELT_FEATS = ["gdelt_tone_72h", "gdelt_goldstein_72h", "gdelt_us_tone_72h",
               "gdelt_tone_chg", "gdelt_volume_ratio"]
MACRO_FEATS = VIX_FEATS + GDELT_FEATS


#helper function to compute ratio, returning NaN when denominator is zero or negative
def _ratio(num, den):
    den = np.where(den > 0, den, np.nan)
    return num / den


#helper function to get the signal bar close time for each setup, used to align macro data with setups
def _bar_close(setups):
    return pd.DatetimeIndex(pd.to_datetime(setups["signal_bar_end"], utc=True))


#helper function to create an empty result dataframe with the same structure as the macro features, used when no data is available
def _empty_result(setups, feats, time_col):
    out = pd.DataFrame({"setup_id": setups["setup_id"].to_numpy()})
    for c in feats:
        out[c] = np.nan
    out[time_col] = pd.to_datetime([pd.NaT] * len(out), utc=True)
    return out


#prepares the vix data for feature building, aligning the index and adding an "available_at" column indicating when the data is available for use
def prep_vix(vix):
    d = vix[["Close"]].copy()
    naive = pd.DatetimeIndex(d.index).astype("datetime64[ns]")
    d.index = naive
    d.index.name = "date"
    d["available_at"] = (naive + pd.Timedelta(days=1)).tz_localize(TZ)
    return d


##prepares the gdelt data for feature building, aligning the index and adding an "available_at" column indicating when the data is available for use
def prep_gdelt(hourly):
    d = hourly.copy()
    d.index = pd.DatetimeIndex(d.index).astype(UTC_NS)
    d.index.name = "hour_utc"
    d["available_at"] = d.index + pd.Timedelta(hours=1)
    return d


#builds the vix features for each setup, merging the vix data with the setups based on the signal date and computing additional features such as 5-day change and comparison to 20-day moving average
def build_vix_features(data, setups):
    if data.empty:
        return _empty_result(setups, VIX_FEATS, "vix_data_time")
    end = _bar_close(setups)
    sig_date = end.tz_convert(TZ).tz_localize(None).normalize().astype("datetime64[ns]")

    v = pd.DataFrame({
        "vix_asof_date": pd.DatetimeIndex(data.index).astype("datetime64[ns]"),
        "vix_close": data["Close"].to_numpy(dtype=float),
    }).sort_values("vix_asof_date")
    v["vix_chg_5d"] = v["vix_close"] / v["vix_close"].shift(5) - 1
    v["vix_vs_ma20"] = v["vix_close"] / v["vix_close"].rolling(20, min_periods=20).mean() - 1

    left = pd.DataFrame({"setup_id": setups["setup_id"].to_numpy(), "signal_date": sig_date,
                         "_row": np.arange(len(end))}).sort_values("signal_date")
    merged = pd.merge_asof(left, v, left_on="signal_date", right_on="vix_asof_date",
                           allow_exact_matches=False, tolerance=pd.Timedelta(days=7))
    merged = merged.sort_values("_row").reset_index(drop=True)
    merged["vix_data_time"] = (merged["vix_asof_date"] + pd.Timedelta(days=1)).dt.tz_localize(TZ)
    return merged[["setup_id"] + VIX_FEATS + ["vix_data_time"]]


##builds the gdelt features for each setup, merging the gdelt data with the setups based on the signal date and computing additional features such as 72-hour and 24-hour rolling sums
def build_gdelt_features(data, setups, window_hours=72, recent_hours=24):
    if data.empty:
        return _empty_result(setups, GDELT_FEATS + ["gdelt_hours_covered"], "gdelt_data_time")
    end = _bar_close(setups).astype(UTC_NS)
    hourly = data.copy()
    hourly.index = pd.DatetimeIndex(hourly.index).astype(UTC_NS)

    # newest bucket b with b + 1h < close:  b = ceil(close - 1h) - 1h
    last_bucket = ((end - pd.Timedelta(hours=1)).ceil("h") - pd.Timedelta(hours=1)).astype(UTC_NS)
    grid_end = max(hourly.index.max(), last_bucket.max())
    grid = pd.date_range(hourly.index.min(), grid_end, freq="h").astype(UTC_NS)

    g = hourly[SUM_COLS].reindex(grid)
    covered = g["mentions"].notna().astype(float)  # 1 if gdelt had any data that hour
    g = g.fillna(0.0)

    W = g.rolling(window_hours, min_periods=1).sum().reindex(last_bucket)   # last 72 buckets
    R = g.rolling(recent_hours, min_periods=1).sum().reindex(last_bucket)   # last 24 buckets
    cov = covered.rolling(window_hours, min_periods=1).sum().reindex(last_bucket)

    tone_72 = _ratio(W["tone_x_m"].to_numpy(), W["mentions"].to_numpy())
    tone_24 = _ratio(R["tone_x_m"].to_numpy(), R["mentions"].to_numpy())
    older_m = (W["mentions"] - R["mentions"]).to_numpy()  # the 48h before the recent 24h
    tone_old = _ratio((W["tone_x_m"] - R["tone_x_m"]).to_numpy(), older_m)
    recent_rate = R["mentions"].to_numpy() / recent_hours
    older_rate = older_m / (window_hours - recent_hours)

    return pd.DataFrame({
        "setup_id": setups["setup_id"].to_numpy(),
        "gdelt_tone_72h": tone_72,
        "gdelt_goldstein_72h": _ratio(W["gold_x_m"].to_numpy(), W["mentions"].to_numpy()),
        "gdelt_us_tone_72h": _ratio(W["us_tone_x_m"].to_numpy(), W["us_mentions"].to_numpy()),
        "gdelt_tone_chg": tone_24 - tone_old,                   # last 24h tone minus the 48h before it
        "gdelt_volume_ratio": _ratio(recent_rate, older_rate),  # news intensity: last 24h vs before
        "gdelt_hours_covered": cov.to_numpy(),                  # data quality: hours (of 72) gdelt has
        "gdelt_data_time": pd.Series(last_bucket + pd.Timedelta(hours=1)),  # newest data used; < signal_bar_end
    })


#builds the combined macro features for each setup, merging the vix and gdelt features based on setup_id
def build_macro_features(vix_data, gdelt_data, setups, window_hours=72, recent_hours=24):
    v = build_vix_features(vix_data, setups)
    g = build_gdelt_features(gdelt_data, setups, window_hours, recent_hours)
    return v.merge(g, on="setup_id", how="left")
