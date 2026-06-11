import argparse
import json
import logging
import os
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


def _prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in AUDIO_FEATURES + ["genre"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[AUDIO_FEATURES].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    y = df["genre"].astype(str)
    return X, y


def _split_data(
    X: pd.DataFrame,
    y_encoded,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    counts = pd.Series(y_encoded).value_counts()
    stratify = y_encoded if counts.min() >= 2 else None

    return train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )


def _classification_metrics(y_true, y_pred) -> Dict[str, float]:
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


def train(data_path: str, params: dict) -> None:
    """Train Logistic Regression and XGBoost models, then log them to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    logger.info("Using MLflow tracking URI: %s", tracking_uri)
    logger.info("Loading training data from %s", data_path)

    df = pd.read_csv(data_path)
    X, y = _prepare_training_data(df)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logger.info("Features shape: %s", X.shape)
    logger.info("Classes: %s", list(label_encoder.classes_))

    X_train, X_valid, y_train, y_valid = _split_data(X, y_encoded)

    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoder, "models/label_encoder.joblib")

    train_params = params.get("train", {})
    run_summaries = []

    for model_name, model_params in train_params.items():
        model_params = dict(model_params or {})
        logger.info("Training model: %s", model_name)

        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_param("model", model_name)
            mlflow.log_params(model_params)
            mlflow.log_param("features", ",".join(AUDIO_FEATURES))

            if model_name == "logistic_regression":
                classifier = LogisticRegression(**model_params)
                model = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("classifier", classifier),
                    ]
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)

                metrics = _classification_metrics(y_valid, y_pred)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=X_valid.head(3),
                )

            elif model_name == "xgboost":
                model_params.setdefault("objective", "multi:softprob")
                model_params.setdefault("eval_metric", "mlogloss")
                model_params.setdefault("random_state", 42)
                model_params.setdefault("n_jobs", -1)
                model_params.setdefault("tree_method", "hist")

                model = xgb.XGBClassifier(**model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)

                metrics = _classification_metrics(y_valid, y_pred)
                mlflow.log_metrics(metrics)
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    input_example=X_valid.head(3),
                )

            else:
                logger.warning("Skipping unsupported model type: %s", model_name)
                continue

            local_model_path = f"models/{model_name}.joblib"
            joblib.dump(model, local_model_path)

            summary = {
                "run_id": run.info.run_id,
                "model": model_name,
                **metrics,
            }
            run_summaries.append(summary)

            logger.info(
                "Finished %s | accuracy=%.4f | f1_macro=%.4f",
                model_name,
                metrics["accuracy"],
                metrics["f1_macro"],
            )

    summary_path = "models/training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "features": AUDIO_FEATURES,
                "classes": list(label_encoder.classes_),
                "runs": run_summaries,
            },
            f,
            indent=2,
        )

    logger.info("Saved training summary to %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r", encoding="utf-8") as f:
        loaded_params = yaml.safe_load(f)

    train(args.data_path, loaded_params)
