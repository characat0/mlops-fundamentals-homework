# Preprocessors

This directory contains the preprocessor artifacts produced by
`data_pipeline/src/train.py` and consumed by the API at inference time:

- `label_encoder.joblib` — maps each genre name to an integer class index
  (and back), fitted on the training set.
- `scaler.joblib` — `StandardScaler` fitted on the training features.
  Used by the `logistic_regression` model; `xgboost` skips it (handled
  internally by the tree algorithm).
- `feature_order.joblib` — the exact list of 12 audio feature names in
  the order the model was trained on. The API uses this to extract
  values from the Pydantic payload in the correct order.

## How to refresh

After re-training (`dvc repro` in `data_pipeline/`), copy the new
artifacts into this directory:

```bash
cp ../data_pipeline/models/*.joblib .
```

The Dockerfile bakes whatever is here at build time into the image, so
make sure the files match the `@champion` model in MLflow.
