import argparse
import os
import mlflow
import yaml
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import xgboost as xgb
import joblib

# Columnas de audio que el modelo usa como entrada
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(data_path: str, params: dict):
    """
    Train multiple genre classification models and log them to MLflow.

    **IMPORTANT: This is an intentionally incomplete skeleton for students to implement.**
    Students are expected to complete the TODO sections below to build a complete
    machine learning training pipeline with proper feature engineering, preprocessing,
    model training, and MLflow logging.

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

    # Seleccion explicita de audio features.
    X = df[AUDIO_FEATURES]
    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Encode del target: genre (texto) -> enteros 0-9
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(f"Clases ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    # Tracking al servidor MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    train_params = params.get("train", {})
    logger.info(f"Training {len(train_params)} model types...")

    os.makedirs("models", exist_ok=True)
    signature = infer_signature(X, y_encoded)

    for model_name, model_params in train_params.items():
        # LogReg necesita escalado
        if model_name == "logistic_regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**model_params)),
            ])
            log_model = mlflow.sklearn.log_model
        elif model_name == "xgboost":
            model = xgb.XGBClassifier(**model_params)
            log_model = mlflow.xgboost.log_model
        else:
            logger.warning(f"Modelo no soportado: {model_name}, se omite")
            continue

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(model_params)
            model.fit(X, y_encoded)

            accuracy = accuracy_score(y_encoded, model.predict(X))
            mlflow.log_metric("accuracy", accuracy)
            logger.info(f"{model_name} accuracy: {accuracy:.4f}")

            # artifact path "model" -> evaluate.py arma runs:/{id}/model
            log_model(model, name="model", signature=signature)
            joblib.dump(model, f"models/{model_name}.joblib")

    joblib.dump(label_encoder, "models/label_encoder.joblib")
    logger.info("Entrenamiento completo. Modelos en models/ y MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
