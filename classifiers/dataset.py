import numpy as np
import pandas as pd
from common.config import PROCESSED_DIR


def add_basic_features(setups): # scale versions of the detector cols so diff priced tickers compare
    d = setups.copy()
    d["atr_pct"] = d["atr"]/d["close"]
    d["dist_ema"] = d["close"] / d["ema_fast"] - 1  # how far above ema price is
    d["dist_sma"] = d["close"] / d["sma_trend"] - 1
    d["trend_gap"] = d["ema_fast"] / d["sma_trend"] - 1  # strength of uptrend
    d["afternoon"] = (pd.to_datetime(d["signal_time"]).dt.hour >= 13).astype(int)  # 13:30 vs 9:30 bar
    return d


def add_macro_features(setups, group): #joins macro_g{group}.parquet (scripts.build_macro_features) on setup_id
    path = PROCESSED_DIR / f"macro_g{group}.parquet"
    if not path.exists():  #macro pipeline not run yet
        return setups
    m = pd.read_parquet(path)
    m = m.drop(columns=[c for c in m.columns if c.endswith("_data_time")])  #cols used only for leakage checks dropped
    return setups.merge(m, on="setup_id", how="left")


def add_relative_features(setups, group): # joins rel_g{group}.parquet (scripts.build_relative_features), same as macro
    path = PROCESSED_DIR / f"rel_g{group}.parquet"
    if not path.exists():
        return setups
    return setups.merge(pd.read_parquet(path).drop(columns="rel_data_time"), on="setup_id", how="left")


def clean_dataset(d, feats=None): # infs to NaN, dupes out. NaNs stay, xgb/lgbm handle them fine
    d = d.drop_duplicates("setup_id", keep="last").copy()
    feats = feats or [c for c in d.columns if pd.api.types.is_float_dtype(d[c])]
    d[feats] = d[feats].replace([np.inf, -np.inf], np.nan)
    return d


def load_dataset(group="all"): # labeled setups w oldest first
    d = pd.read_parquet(PROCESSED_DIR / f"setups_g{group}.parquet")
    d= d[d["label"].notna()].copy()
    d["label"] = d["label"].astype(int)
    d = add_macro_features(d, group)
    d = add_relative_features(d, group)
    d = clean_dataset(add_basic_features(d))
    return d.sort_values("signal_time").reset_index(drop=True)
