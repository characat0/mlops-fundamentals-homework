import argparse
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import yaml
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
#import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms"
]


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

    # X = df.drop(["genre", "year"], axis=1, errors='ignore')
    feature_cols = [f for f in AUDIO_FEATURES if f in df.columns]
    logger.info(f"Using features: {feature_cols}")

    X = df[feature_cols]

    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Unique genres: {sorted(y.unique())}")

    # ENCODING:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info(f"Encoded {len(le.classes_)} genre classes: {list(le.classes_)}")

    # SCALING:

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("spotify-genre-classification")

    # Get hyperparameters
    train_params = params.get("train", {})

    logger.info(f"Training {len(train_params)} model types...")

    # MODEL TRAINING LOOP STRUCTURE:
    for model_name, model_params in train_params.items():
        logger.info(f"Training model: {model_name} with params: {model_params}")

        if model_name == "logistic_regression":
            model = LogisticRegression(**model_params)
            X_to_use = X_scaled
        elif model_name == "xgboost":
            # Map genre classes to contiguous 0..N-1 range (required by XGBoost)
            model = xgb.XGBClassifier(
                **model_params,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0
            )
            X_to_use = X.values
        else:
            logger.warning(f"Unknown model type: {model_name}, skipping")
            continue

        with mlflow.start_run(run_name=model_name):
            # Log hyperparameters
            mlflow.log_params(model_params)
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("n_classes", len(le.classes_))

            # Train the model
            logger.info(f"Fitting {model_name}...")
            model.fit(X_to_use, y_encoded)

            # Evaluate on training data
            y_pred = model.predict(X_to_use)
            accuracy = accuracy_score(y_encoded, y_pred)
            precision = precision_score(
                y_encoded, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_encoded, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(
                y_encoded, y_pred, average="weighted", zero_division=0
            )

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            logger.info(
                f"{model_name}: accuracy={accuracy:.4f}, "
                f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
            )

            # Log model artifact
            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            logger.info(f"Model {model_name} logged to MLflow successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
