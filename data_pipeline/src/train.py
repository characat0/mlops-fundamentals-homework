import argparse
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import yaml
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio features used as model inputs (everything else is metadata).
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]


def train(data_path: str, params: dict):
    """
    Train multiple genre classification models and log them to MLflow.

    Trains a LogisticRegression (on standardized features) and an XGBoost
    classifier (on raw features), logging hyperparameters, metrics, and the
    fitted model artifact to a separate MLflow run for each model type.

    Args:
        data_path: Path to training CSV file (from data pipeline)
        params: Dictionary with hyperparameters from params.yaml
    """
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    # Point MLflow at the tracking server (env var, falls back to localhost).
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("spotify-genre-classification")
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # FEATURE SELECTION:
    # Keep only the numeric audio features — drop all metadata/text columns
    # (id, name, artists, lyrics, popularity, year, ...). Selecting an explicit
    # list is safer than drop() because it never leaks string columns into the
    # model (sklearn/XGBoost would crash on them).
    feature_cols = [c for c in AUDIO_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df["genre"]

    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # ENCODING:
    # LabelEncoder maps the genre strings to integers 0..n-1, sorted
    # alphabetically. The serving layer maps the predicted index back to a
    # genre name using the same sorted order.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(f"Encoded {len(label_encoder.classes_)} genres: {list(label_encoder.classes_)}")

    # SCALING:
    # LogisticRegression needs standardized features; XGBoost is tree-based and
    # scale-invariant, so it trains on the raw features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Persist preprocessing artifacts locally. DVC tracks the models/ directory
    # as the output of the train stage (see dvc.yaml), so it must exist after
    # this script runs — the authoritative model registry still lives in MLflow.
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.joblib"))

    # Get hyperparameters
    train_params = params.get("train", {})

    logger.info(f"Training {len(train_params)} model types...")

    for model_name, model_params in train_params.items():
        # Pick model class and the matching feature representation.
        if model_name == "logistic_regression":
            model = LogisticRegression(**model_params)
            X_to_use = X_scaled
        elif model_name == "xgboost":
            model = xgb.XGBClassifier(**model_params)
            X_to_use = X
        else:
            logger.warning(f"Unknown model '{model_name}', skipping.")
            continue

        with mlflow.start_run(run_name=model_name):
            # Log hyperparameters + a 'model' tag (evaluate.py reads it back).
            mlflow.log_params(model_params)
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_features", X_to_use.shape[1])

            model.fit(X_to_use, y_encoded)

            y_pred = model.predict(X_to_use)
            accuracy = accuracy_score(y_encoded, y_pred)
            precision = precision_score(
                y_encoded, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_encoded, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(y_encoded, y_pred, average="weighted", zero_division=0)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Save the model artifact under "model" so evaluate.py can build
            # the URI runs:/<run_id>/model.
            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            # Also persist locally for the DVC train-stage output.
            joblib.dump(model, os.path.join(models_dir, f"{model_name}.joblib"))

            logger.info(
                f"[{model_name}] accuracy={accuracy:.4f} f1={f1:.4f} logged to MLflow"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
