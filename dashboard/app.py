import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit runs from dashboard/ so repo root isnt on path

import pandas as pd
import streamlit as st

from common.config import PROCESSED_DIR, load_strategy, load_yaml
from dashboard.probs import at_threshold, in_group, load_model, pending_setups, score_setups, test_predictions

# run train_xgb first
# streamlit run dashboard/app.py

MODEL_GROUP = "all"  # only trained one model so far, on every ticker


@st.cache_data
def load_all():
    mcfg = load_yaml("model.yaml")
    model = load_model(MODEL_GROUP)
    pending = score_setups(pending_setups(MODEL_GROUP), model, mcfg["features"])
    with open(PROCESSED_DIR / f"xgb_results_g{MODEL_GROUP}.json", encoding="utf-8") as f:
        results = json.load(f)
    return mcfg, pending, test_predictions(MODEL_GROUP), results


def pct(x):
    return "" if pd.isna(x) else f"{x:.1%}"


def score(x): # not a % bc scores arent real chances of winning
    return f"{x:.2f}"


def show_time(s): # tz aware -> plain eastern time string
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d %H:%M")


st.set_page_config(page_title="Setup scores", layout="wide")
mcfg, pending, test, results = load_all()
lab = load_strategy()["labels"]

st.title("Setup scores", anchor=False)
st.write(f"XGBoost score for each moving average pullback setup, higher means the model ranks it better. A win means price hits the "
         f"{lab['target_atr']:g} ATR target before the {lab['stop_atr']:g} ATR stop, within {lab['horizon_bars']} four hour bars. "
         f"A score is not the chance of winning: wins are weighted up in training, so scores run high. "
         f"Use the actual win rates below instead.")

with st.sidebar:
    group = st.selectbox("Ticker group", ["all", "1", "2", "3"])
    threshold = st.slider("Take setups with score at least", 0.0, 1.0, float(mcfg["threshold"]), 0.01)
    st.caption("Moving the threshold changes which setups count as picked, in both tables.")

# open setups
st.header("Open setups", anchor=False)
p = in_group(pending, group)
if p.empty:
    st.write("No open setups for this group right now.")
else:
    st.dataframe(pd.DataFrame({
        "Ticker": p["ticker"],
        "Signal (ET)": show_time(p["signal_time"]),
        "Status": p["stop"].isna().map({True: "waiting for entry", False: "in trade"}),  # stop set at entry bar
        "Close": p["close"].round(2),
        "Stop": p["stop"].round(2),
        "Target": p["target"].round(2),
        "Score": p["prob"].map(score),
        "Take": (p["prob"] >= threshold).map({True: "yes", False: "no"}),
    }), hide_index=True, width="stretch")
    st.caption("Open means there's no outcome yet. Waiting for entry means the next bar hasn't opened, so the stop and "
               "target aren't set. In trade means it hasn't hit the stop, the target or the time limit yet.")

# test period
t = in_group(test, group)
s = at_threshold(t, threshold)
st.header("How the scores did on the test period", anchor=False)
st.write(f"{pd.to_datetime(t['signal_time']).min():%b %d, %Y} to {pd.to_datetime(t['signal_time']).max():%b %d, %Y}. "
         f"The model never saw these setups during training or tuning.")
st.table(pd.DataFrame({
    "Setups": [s["setups"], s["picked"]],
    "Win rate": [pct(s["win_rate_all"]), pct(s["win_rate_picked"])],
    "Avg R per trade": [f"{s['avg_r_all']:.2f}", "" if pd.isna(s["avg_r_picked"]) else f"{s['avg_r_picked']:.2f}"],
}, index=["Every setup", f"Score at least {threshold:.2f}"]))
st.write(f"Test ROC-AUC was {results['test']['roc_auc']:.2f}, close to the 0.5 you'd get by guessing, "
         f"so read these scores as rough."
         + ("" if "vix_close" in mcfg["features"] else " Macro features aren't in the model yet."))

with st.expander("Every test setup"):
    t = t.sort_values("prob", ascending=False)  # sort before pct() or 9% lands above 57%
    st.dataframe(pd.DataFrame({
        "Ticker": t["ticker"],
        "Signal (ET)": show_time(t["signal_time"]),
        "Score": t["prob"].map(score),
        "Picked": (t["prob"] >= threshold).map({True: "yes", False: "no"}),
        "Outcome": t["label"].map({1: "win", 0: "loss"}),
        "R": t["ret_R"].round(2),
    }), hide_index=True, width="stretch")
