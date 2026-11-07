import pandas as pd

from backtest.metrics import max_drawdown, sharpe
from common.config import PROCESSED_DIR, load_yaml
from features.technical import TECH_FEATS

# compare detector/labeler output across the 3 ticker groups, are some sectors way diff?
# run build_setups for group 1, 2 and 3 first


def group_stats(d, n_tickers):
    done = d[d["label"].notna()]
    r = done["ret_R"].astype(float)
    bar = done["barrier"].value_counts(normalize=True)
    return {
        "tickers": n_tickers,
        "tickers_w_setups": d["ticker"].nunique(),
        "setups": len(d),
        "pending": int(d["label"].isna().sum()),
        "setups_per_ticker": round(len(d) / n_tickers, 1),
        "win_rate": round(done["label"].mean(), 3),  # target hit (label), not just>0R
        "avg_R": round(r.mean(), 3),
        "total_R": round(r.sum(), 1),
        "sharpe": round(sharpe(done), 2),
        "max_dd_R": round(max_drawdown(done), 1),
        "pct_tp": round(bar.get("tp", 0), 3),
        "pct_sl": round(bar.get("sl", 0), 3),
        "pct_time": round(bar.get("time", 0), 3),
        "avg_bars_held": round(done["bars_held"].astype(float).mean(), 1),
        "same_bar_ties": int(done["same_bar"].fillna(False).astype(bool).sum()),
        "first": f"{d['signal_time'].min():%Y-%m-%d}",
        "last": f"{d['signal_time'].max():%Y-%m-%d}",
    }


def main():
    groups = load_yaml("tickers.yaml")["groups"]
    rows, feats = {}, {}
    for g, tickers in groups.items():
        path = PROCESSED_DIR / f"setups_g{g}.parquet"
        if not path.exists():
            print(f"skipping group {g}, run python -m scripts.build_setups --group {g}")
            continue
        d = pd.read_parquet(path)
        rows[f"g{g}"] = group_stats(d, len(tickers))
        have = [c for c in ["rsi"] + TECH_FEATS if c in d.columns]
        feats[f"g{g}"] = d[have].mean()
    if not rows:
        raise SystemExit("no setups files found")

    table = pd.DataFrame(rows)
    print("detector + labels by group:")
    print(table.to_string())
    print("\navg feature value at signal by group (big gaps = model might just learn the sector):")
    print(pd.DataFrame(feats).round(4).to_string())

    # per ticker so we can spot ones dragging a group down
    per = []
    for g in groups:
        path = PROCESSED_DIR / f"setups_g{g}.parquet"
        if path.exists():
            d = pd.read_parquet(path)
            t = d[d["label"].notna()].groupby("ticker").agg(setups=("setup_id", "size"),
                win_rate=("label", "mean"), avg_R=("ret_R", "mean"), total_R=("ret_R", "sum"))
            per.append(t.assign(group=g))
    per = pd.concat(per).sort_values("total_R")
    print("\nworst 5 / best 5 tickers by total R:")
    print(pd.concat([per.head(5), per.tail(5)]).round(3).to_string())

    table.to_csv(PROCESSED_DIR / "group_comparison.csv")
    per.round(4).to_csv(PROCESSED_DIR / "group_comparison_tickers.csv")
    print(f"\nsaved to {PROCESSED_DIR / 'group_comparison.csv'} and group_comparison_tickers.csv")


if __name__ == "__main__":
    main()
