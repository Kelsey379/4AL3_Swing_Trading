import argparse

import pandas as pd

from backtest.baselines import buy_and_hold, daily_closes, equal_weight_hold, unfiltered_detector
from common.config import PROCESSED_DIR, load_yaml
from data_collection.ohlcv import TZ, drop_incomplete, load_cached_hourly, resample_4h

# baselines: buy and hold spy/qqq + the groups own tickers, and the detector w/ no model filter
# needs pull_ohlcv --benchmarks and build_setups first
# python -m scripts.run_baselines --group all
# python -m scripts.run_baselines --group all --test-window   <- same dates as xgb test set (run train_xgb first)


def closes_for(tickers, now):
    out = {}
    for t in tickers:
        h = load_cached_hourly(t)
        if h.empty:
            print(f"{t} not cached, run python -m scripts.pull_ohlcv --benchmarks")
            continue
        out[t] = daily_closes(drop_incomplete(resample_4h(h), now))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="1, 2, 3 or all")
    ap.add_argument("--test-window", action="store_true", help="only the xgb test period")
    ap.add_argument("--risk", type=float, default=0.01, help="fraction of acct risked per trade (1R)")
    args = ap.parse_args()

    setups = pd.read_parquet(PROCESSED_DIR / f"setups_g{args.group}.parquet")
    st = pd.to_datetime(setups["signal_time"]).dt.tz_localize(None)
    start, end = st.min().normalize(), st.max().normalize()

    preds = None
    if args.test_window:
        p = PROCESSED_DIR / f"xgb_test_preds_g{args.group}.parquet"
        if not p.exists():
            raise SystemExit(f"no {p.name}, run python -m scripts.train_xgb --group {args.group}")
        preds = pd.read_parquet(p)
        pt = pd.to_datetime(preds["signal_time"]).dt.tz_localize(None)
        start, end = pt.min().normalize(), pt.max().normalize()
    print(f"window {start:%Y-%m-%d} to {end:%Y-%m-%d}, risk {args.risk:.1%} per trade\n")

    now = pd.Timestamp.now(tz=TZ)
    groups = load_yaml("tickers.yaml")["groups"]
    tickers = setups["ticker"].unique().tolist() if args.group == "all" else groups[int(args.group)]
    rows = {}
    for b, c in closes_for(load_yaml("tickers.yaml")["benchmarks"], now).items():
        rows[f"buy_hold_{b}"] = buy_and_hold(c, start, end)
    rows[f"buy_hold_eqw_g{args.group}"] = equal_weight_hold(closes_for(tickers, now), start, end)
    rows["detector_unfiltered"] = unfiltered_detector(setups, start, end, args.risk)

    if preds is not None: # side by side w/ what the model picked, threshold from model.yaml
        thr = load_yaml("model.yaml")["threshold"]
        picked = setups[setups["setup_id"].isin(preds.loc[preds["prob"] >= thr, "setup_id"])]
        rows[f"detector_xgb_p>={thr}"] = unfiltered_detector(picked, start, end, args.risk) if len(picked) else {"trades": 0}

    table = pd.DataFrame(rows)
    print(table.to_string())
    name = f"baselines_g{args.group}{'_test' if args.test_window else ''}.csv"
    table.to_csv(PROCESSED_DIR / name)
    print(f"\nsaved to {PROCESSED_DIR / name}")
    print("total_return/cagr/max_dd are fractions (0.12 = 12%). R_ rows are in R, only for detector rows")


if __name__ == "__main__":
    main()
