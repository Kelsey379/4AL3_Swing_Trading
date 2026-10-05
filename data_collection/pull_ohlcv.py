import argparse
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from common.config import RAW_OHLCV_DIR, load_strategy, tickers_for

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
RESAMPLE_RULES = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def pull_hourly(ticker, start):
    df = yf.Ticker(ticker).history(start=start, interval="1h", auto_adjust=True, prepost=False)
    if df.empty:
        return df
    df = df[COLUMNS].dropna(how="all")
    df.index = df.index.tz_convert("America/New_York")
    return df


def to_4h(hourly):
    day = hourly.index.normalize()
    half = (hourly.index.hour >= 13).astype(int)
    bars = (
        hourly.assign(Datetime=hourly.index)
        .groupby([day, half])
        .agg({**RESAMPLE_RULES, "Datetime": "first"})
        .set_index("Datetime")
    )
    return bars[bars["Volume"] > 0]


def pull_group(group):
    data_cfg = load_strategy()["data"]
    start = date.today() - timedelta(days=data_cfg["lookback_days"] + data_cfg["warmup_days"])
    RAW_OHLCV_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for ticker in tickers_for(group):
        hourly = pull_hourly(ticker, start)
        if hourly.empty:
            failed.append(ticker)
            continue
        bars = to_4h(hourly)
        bars.to_parquet(RAW_OHLCV_DIR / f"{ticker}_4h.parquet")
        print(f"{ticker}: {len(bars)} bars {bars.index[0].date()} to {bars.index[-1].date()}")

    if failed:
        print("no data for:", ", ".join(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="3")
    pull_group(parser.parse_args().group)
