import argparse
import os
import sys
import joblib
import logging

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in AUDIO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Expected audio features missing from input: {missing}")
    return df[AUDIO_FEATURES].copy()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _build_model(model_name: str, model_params: dict):
    if model_name == "logistic_regression":
        return LogisticRegression(**model_params)
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            **model_params, use_label_encoder=False, eval_metric="mlogloss"
        )
    raise ValueError(f"Unknown model_name: {model_name!r}")


def _log_model(model, model_name: str) -> None:
    if model_name == "logistic_regression":
        mlflow.sklearn.log_model(model, artifact_path="model")
    elif model_name == "xgboost":
        mlflow.xgboost.log_model(model, artifact_path="model")


def train(data_path: str, params: dict, tracking_uri: str, models_dir: str) -> None:
    """
    Train multiple genre classification models and log them to MLflow.
    Splits the data into train/test (stratified, random_state=42) and
    logs both train and test metrics. The test split serves as an
    honest in-distribution holdout; the prod_sim.csv split (years
    >2010) is evaluated separately in evaluate.py to quantify
    temporal drift.
    Args:
        data_path: Path to training CSV file (from data pipeline).
        params: Dictionary with hyperparameters from params.yaml.
        tracking_uri: MLflow tracking server URI.
        models_dir: Local directory where the LabelEncoder and StandardScaler
            are persisted so evaluate.py / the API can reuse them.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("spotify-genre-classifier")
    logger.info(f"Using MLflow tracking URI: {tracking_uri}")
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Raw shape: {df.shape}")
    X = _select_features(df)
    y_raw = df["genre"]
    logger.info(f"Features shape: {X.shape}, Target shape: {y_raw.shape}")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    logger.info(
        f"Encoded {len(label_encoder.classes_)} genre classes: "
        f"{list(label_encoder.classes_)}"
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(AUDIO_FEATURES, os.path.join(models_dir, "feature_order.joblib"))
    logger.info(f"Persisted encoder, scaler, and feature order to {models_dir}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    X_train_raw, X_test_raw, _, _ = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    logger.info(
        f"Train/test split: train={len(X_train)}, test={len(X_test)} "
        f"(stratified, random_state={RANDOM_STATE}, test_size={TEST_SIZE})"
    )
    train_params = params.get("train", {})
    logger.info(f"Training {len(train_params)} model type(s): {list(train_params)}")
    for model_name, model_params in train_params.items():
        logger.info(f"--- Training {model_name} ---")
        model = _build_model(model_name, model_params)
        is_tree_based = model_name == "xgboost"
        X_tr = X_train_raw if is_tree_based else X_train
        X_te = X_test_raw if is_tree_based else X_test
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("model_type", model_name)
            mlflow.log_params(model_params)
            mlflow.log_params(
                {
                    "random_state": RANDOM_STATE,
                    "test_size": TEST_SIZE,
                    "stratified": True,
                }
            )
            model.fit(X_tr, y_train)
            train_metrics = _compute_metrics(y_train, model.predict(X_tr))
            test_metrics = _compute_metrics(y_test, model.predict(X_te))
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            # Aliases: keep the bare "accuracy" key for backward-compat
            # with the rubric (search_runs(order_by="metrics.accuracy DESC")
            # in evaluate.py) — it points to the test number.
            mlflow.log_metrics(
                {
                    "accuracy": test_metrics["accuracy"],
                    "precision_macro": test_metrics["precision_macro"],
                    "recall_macro": test_metrics["recall_macro"],
                    "f1_macro": test_metrics["f1_macro"],
                }
            )
            logger.info(f"{model_name} train metrics: {train_metrics}")
            logger.info(f"{model_name} test  metrics: {test_metrics}")
            _log_model(model, model_name)
            logger.info(f"Logged {model_name} model artifact to MLflow")
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()
    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)
    train(args.data_path, params, args.tracking_uri, args.models_dir)
