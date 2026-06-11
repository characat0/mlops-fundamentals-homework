import argparse
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms"
]


def _validate_columns(df: pd.DataFrame) -> None:
    required_columns = AUDIO_FEATURES + ["genre"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")


def train(data_path: str, params: dict):
    """
    Train Logistic Regression and XGBoost genre classifiers and log them to MLflow.

    Args:
        data_path: Path to training CSV file (from data pipeline)
        params: Dictionary with hyperparameters from params.yaml
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    _validate_columns(df)

    df = df.dropna(subset=AUDIO_FEATURES + ["genre"]).copy()
    X = df[AUDIO_FEATURES]
    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    os.makedirs("models", exist_ok=True)
    label_encoder_path = os.path.join("models", "label_encoder.joblib")
    joblib.dump(label_encoder, label_encoder_path)

    train_params = params.get("train", {})
    logger.info(f"Training {len(train_params)} model types...")

    for model_name, model_params in train_params.items():
        run_params = dict(model_params)

        if model_name == "logistic_regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**run_params))
            ])
            log_model = mlflow.sklearn.log_model
        elif model_name == "xgboost":
            run_params.setdefault("eval_metric", "mlogloss")
            model = xgb.XGBClassifier(**run_params)
            log_model = mlflow.xgboost.log_model
        else:
            logger.warning(f"Skipping unsupported model type: {model_name}")
            continue

        logger.info(f"Training {model_name}...")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_params(run_params)

            model.fit(X, y_encoded)
            y_pred = model.predict(X)

            metrics = {
                "accuracy": accuracy_score(y_encoded, y_pred),
                "precision_weighted": precision_score(
                    y_encoded, y_pred, average="weighted", zero_division=0
                ),
                "recall_weighted": recall_score(
                    y_encoded, y_pred, average="weighted", zero_division=0
                ),
                "f1_weighted": f1_score(
                    y_encoded, y_pred, average="weighted", zero_division=0
                ),
            }
            mlflow.log_metrics(metrics)

            log_model(model, artifact_path="model")
            mlflow.log_artifact(label_encoder_path, artifact_path="model")

            local_model_path = os.path.join("models", f"{model_name}.joblib")
            joblib.dump(model, local_model_path)
            logger.info(
                f"{model_name} accuracy={metrics['accuracy']:.4f}; "
                f"saved local copy to {local_model_path}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
