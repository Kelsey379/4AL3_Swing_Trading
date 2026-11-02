import numpy as np
import pandas as pd

# triple barrier w/ win (1) = target hit before stop within horizon_bars, everything else = loss (0), pending = not enough future bars yet, label stays NaN

RESULT_COLS = ["entry_time", "entry_price", "stop", "target", "exit_bar", "exit_time", "exit_price",
               "barrier", "bars_held", "ret_R", "same_bar", "label"]


def _resolve_same_bar(hourly, start, end, stop, target): # both barriers in one 4h bar, check 1h bars for which was first
    if hourly is not None and not hourly.empty:
        hrs = hourly[(hourly.index >= start) & (hourly.index < end)]
        for low, high in zip(hrs["Low"], hrs["High"]):
            hit_sl, hit_tp = low <= stop, high >= target
            if hit_sl and hit_tp:
                break  # both  same hour
            if hit_sl:
                return "sl"
            if hit_tp:
                return "tp"
    return "sl"  # cant tell so assume worst


def _walk(bars, s, lc, hourly):
    res = dict.fromkeys(RESULT_COLS)
    i = bars.index.get_loc(s["signal_time"])
    e = i + 1  # enter on next bar open, never signal bar close
    if e >= len(bars):
        return res

    entry = bars["Open"].iat[e]
    risk = lc["stop_atr"] * s["atr"]
    stop, target = entry - risk, entry + lc["target_atr"] * s["atr"]
    res.update(entry_time=bars.index[e], entry_price=entry, stop=stop, target=target)

    def done(j, price, barrier, same_bar=False):
        res.update(exit_bar=bars.index[j], exit_time=bars["bar_end"].iat[j], exit_price=price,
                   barrier=barrier, bars_held=j - e + 1, ret_R=(price - entry) / risk,
                   same_bar=same_bar, label=int(barrier == "tp"))
        return res

    last = e + lc["horizon_bars"] - 1
    for j in range(e, min(last, len(bars) - 1) + 1):
        o, h, l = bars["Open"].iat[j], bars["High"].iat[j], bars["Low"].iat[j]

        # gapped past a barrier overnight, fill at the open not the barrier
        if o <= stop:
            return done(j, o, "sl")
        if o >= target:
            return done(j, o, "tp")

        hit_sl, hit_tp = l <= stop, h >= target
        if hit_sl and hit_tp:
            barrier = _resolve_same_bar(hourly, bars.index[j], bars["bar_end"].iat[j], stop, target)
            return done(j, target if barrier == "tp" else stop, barrier, same_bar=True)
        if hit_sl:
            return done(j, stop, "sl")
        if hit_tp:
            return done(j, target, "tp")

    if last >= len(bars):
        return res  # ran out of data before time limit, still pending
    return done(last, bars["Close"].iat[last], "time")


def label_setups(setups, bars, cfg, hourly=None): # one ticker at a time, hourly only used for same bar ties
    lc = cfg["labels"]
    rows = [{**s, **_walk(bars, s, lc, hourly)} for s in setups.to_dict("records")]
    out = pd.DataFrame(rows, columns=list(setups.columns) + RESULT_COLS)
    out["label"] = out["label"].astype(float)  # NaN = pending
    return out


def drop_overlapping(labeled): # one position per ticker, skip setups while last trade still open
    keep = []
    for _, g in labeled.sort_values("signal_time").groupby("ticker", sort=False):
        in_trade, free_at = False, None  # free_at = when last kept trade closed (NaT if still open)
        for idx, r in g.iterrows():
            if in_trade and (pd.isna(free_at) or r["signal_time"] < free_at):
                continue
            keep.append(idx)
            in_trade, free_at = True, r["exit_time"]
    return labeled.loc[keep].sort_values(["ticker", "signal_time"]).reset_index(drop=True)


def summarize(labeled):
    done = labeled[labeled["label"].notna()]
    return {
        "setups": len(labeled),
        "labeled": len(done),
        "pending": int(labeled["label"].isna().sum()),
        "win_rate": round(done["label"].mean(), 3) if len(done) else np.nan,
        "avg_R": round(done["ret_R"].astype(float).mean(), 3) if len(done) else np.nan,
        "barriers": done["barrier"].value_counts().to_dict(),
        "same_bar_ties": int(done["same_bar"].fillna(False).astype(bool).sum()),
    }
