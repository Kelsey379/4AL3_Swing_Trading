4AL3 Swing Trading

Tests if a model can tell which moving average pullback setups are worth taking. A rule based detector finds long setups on 4h bars for 76 stocks, each one gets a triple barrier label, then XGBoost and LightGBM give each setup a probability of hitting the target.

Setup

1. python 3.13
2. python -m venv .venv
3. .venv\Scripts\activate on windows, source .venv/bin/activate otherwise
4. pip install -r requirements.txt
5. GDELT needs BigQuery access to the swingtrading4al3 project. Put the service account key json in the repo root, copy .env.example to .env and set GOOGLE_APPLICATION_CREDENTIALS to the key path (both are gitignored). Yahoo and VIX don't need a key.

data/raw and data/processed are gitignored, so run the pipeline once or get the files from someone. Run everything from the repo root.

Data

1. python -m scripts.pull_ohlcv --group all --benchmarks
2. python -m scripts.pull_macro (use --dry-run first to see how much BigQuery would scan)
3. python -m scripts.build_setups --group all
4. python -m scripts.build_macro_features --group all
5. python -m scripts.build_dataset --group all

Output goes to data/processed. dataset_gall.parquet has the same rows for every model.

Leakage checks

- python -m scripts.check_technical_leakage --group all
- python -m scripts.check_macro_leakage --group all

Models

- python -m scripts.train_xgb --group all
- python -m scripts.train_lgbm --group all
- python -m scripts.walk_forward_xgb --group all --compare (runs each feature set over the full history)
- python -m scripts.walk_forward_lgbm --group all --compare
- python -m scripts.run_baselines --group all --test-window (run train_xgb first, it uses the xgb test dates)
- python -m scripts.plot_xgb --group all
- python -m scripts.plot_lgbm --group all

Results go to data/processed, plots to data/plots, models to data/models.

Groups

To compare the three ticker groups, run build_setups with --group 1, 2 and 3, then python -m scripts.compare_groups

Dashboard

streamlit run dashboard/app.py (run train_xgb first)

Tests

pytest

Settings are in config/ (strategy.yaml for the setup and label rules, model.yaml for training, tickers.yaml for the groups). docs/setup_spec.md has more detail on the rules and features. The report is in documents/Proposal LaTeX.
