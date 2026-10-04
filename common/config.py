from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT/ "config"
DATA_DIR = ROOT/"data"
RAW_OHLCV_DIR = DATA_DIR /"raw" /"ohlcv"
PROCESSED_DIR = DATA_DIR /"processed"


def load_yaml(name):
    with open(CONFIG_DIR / name,encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_strategy():
    return load_yaml("strategy.yaml")


def tickers_for(group): # group 1, 2, 3, all
    groups = load_yaml("tickers.yaml")["groups"]

    if str(group) == "all":
        return [t for g in groups.values() for t in g]
    
    return list(groups[int(group)])
