# Drift Monitoring

Statistical drift detection (Kolmogorov-Smirnov test) for the Spotify genre
classifier. Compares feature distributions between:

- **Batch mode** (`--mode batch`): `data/train.csv` vs `data/prod_sim.csv`
  (the temporal split produced by `data_pipeline/src/process.py`).
- **Online mode** (`--mode online`): `data/train.csv` vs
  `model_serving/logs/api_requests.jsonl` (live API traffic captured by
  the request-logging middleware).

## Usage

```bash
cd drift_monitoring
uv sync

# Batch drift: pre-2010 vs post-2010
uv run python src/analyze_drift.py \
  --mode batch \
  --train_data ../data_pipeline/data/train.csv \
  --prod_data ../data_pipeline/data/prod_sim.csv \
  --output batch_drift_report.json

# Online drift: training baseline vs live API traffic
uv run python src/analyze_drift.py \
  --mode online \
  --train_data ../data_pipeline/data/train.csv \
  --api_logs ../model_serving/logs/api_requests.jsonl \
  --output online_drift_report.json
```

## Output

Each run writes a JSON report with:

- `status`: `DRIFT_DETECTED` if more than 20% of features drifted, else `NORMAL`
- `features_with_drift`: count of features with `p_value < 0.05`
- `drifted_features`: list of drifted feature names
- `details`: per-feature `{ks_statistic, p_value, drift_detected, train_mean, prod_mean}`
