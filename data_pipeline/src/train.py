import argparse
import mlflow
import yaml
import pandas as pd
import logging
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(data_path: str, params: dict):
    """
    Train multiple genre classification models and log them to MLflow.

    Args:
        data_path: Path to training CSV file (from data pipeline)
        params: Dictionary with hyperparameters from params.yaml

    Implementation Steps:
        1. Load the training data from data_path
        2. Separate features (X) from target (y)
           - Target: 'genre' column (10 classes)
           - Features: audio feature columns
           - Drop metadata columns (id, name, artist, year, popularity, etc.)
        3. Encode genre labels using LabelEncoder
        4. Scale features using StandardScaler
        5. For each model type in params['train']:
           a. Start an MLflow run with run_name
           b. Log parameters from params.yaml
           c. Train the model on scaled X
           d. Calculate accuracy metric
           e. Log metrics to MLflow
           f. Log model artifact with appropriate MLflow function
           g. End the run
    """
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    # FEATURE SELECTION:
    audio_features = [
        "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
    ]
    X = df[audio_features]
    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # ENCODING:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoder, "models/label_encoder.joblib")

    # SCALING:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.joblib")

    # Get hyperparameters
    train_params = params.get("train", {})

    logger.info(f"Training {len(train_params)} model types...")

    for model_name, model_params in train_params.items():
        logger.info(f"Training {model_name}...")

        if model_name == 'logistic_regression':
            model = LogisticRegression(**model_params)
            X_to_use = X_scaled
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(**model_params)
            X_to_use = X
        else:
            logger.warning(f"Unknown model type: {model_name}. Skipping.")
            continue

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", model_name)
            model.fit(X_to_use, y_encoded)
            accuracy = model.score(X_to_use, y_encoded)
            mlflow.log_metric("accuracy", accuracy)

            if model_name == 'logistic_regression':
                mlflow.sklearn.log_model(model, artifact_path="model")
            elif model_name == 'xgboost':
                mlflow.xgboost.log_model(model, artifact_path="model")

            logger.info(f"{model_name} training complete. Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
