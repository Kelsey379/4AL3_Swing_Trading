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


def load_dataset(group="all"): # labeled setups w oldest first
    d = pd.read_parquet(PROCESSED_DIR / f"setups_g{group}.parquet")
    d= d[d["label"].notna()].copy()
    d["label"] = d["label"].astype(int)
    return add_basic_features(d).sort_values("signal_time").reset_index(drop=True)
