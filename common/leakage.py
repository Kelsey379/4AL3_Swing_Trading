import pandas as pd


# for the macro features table needs setup_id + col saying when the data was available
def check_feature_times(setups, feats, time_col, key="setup_id"): # every feature has to be from before its setup's bar closed
    m= setups[[key, "signal_bar_end"]].merge(feats[[key, time_col]], on=key, how="inner")
    bad = m[pd.to_datetime(m[time_col]) >= m["signal_bar_end"]]
    
    if len(bad):
        raise ValueError(f"{len(bad)} feature rows use data from at/after bar close, e.g. {bad[key].iloc[0]}")
    return len(m)
def check_truncation(build_fn, data, setups, cutoffs, time_col): # rebuilding w/ future data cut off can't change older rows
    
    full =build_fn(data, setups).set_index("setup_id").sort_index()
    
    for cut in cutoffs:
        part = build_fn(data[pd.to_datetime(data[time_col]) < cut], setups).set_index("setup_id").sort_index()
        old = setups.loc[setups["signal_bar_end"] <= cut, "setup_id"]
        pd.testing.assert_frame_equal(part.loc[part.index.intersection(old)], full.loc[full.index.intersection(old)],
                                      check_dtype=False)
    
    
    return True
