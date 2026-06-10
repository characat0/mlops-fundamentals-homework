import argparse
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import yaml
import pandas as pd
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import os
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]


def train(data_path: str, params: dict):
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    available_features = [f for f in AUDIO_FEATURES if f in df.columns]
    logger.info(f"Using features: {available_features}")

    X = df[available_features]
    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_params = params.get("train", {})
    logger.info(f"Training {len(train_params)} model types...")

    os.makedirs("models", exist_ok=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("spotify-genre-classification")

    for model_name, model_params in train_params.items():
        logger.info(f"Training {model_name}...")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(model_params)

            if model_name == "logistic_regression":
                model = LogisticRegression(**model_params)
                model.fit(X_scaled, y_encoded)
                y_pred = model.predict(X_scaled)
                accuracy = accuracy_score(y_encoded, y_pred)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.sklearn.log_model(model, artifact_path="model")
                joblib.dump(model, f"models/{model_name}.pkl")

            elif model_name == "xgboost":
                model = xgb.XGBClassifier(**model_params)
                model.fit(X, y_encoded)
                y_pred = model.predict(X)
                accuracy = accuracy_score(y_encoded, y_pred)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.xgboost.log_model(model, artifact_path="model")
                joblib.dump(model, f"models/{model_name}.pkl")

            logger.info(f"{model_name} accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
