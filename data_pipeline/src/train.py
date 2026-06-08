import argparse
import os
import mlflow
from sklearn.metrics import accuracy_score
import yaml
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("models", exist_ok=True)


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

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("spotify-genre-classification")
    logger.info(f"Using MLflow tracking URI: {tracking_uri}")

    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    # FEATURE SELECTION:
    # Students should select features from the Kaggle dataset.
    # Drop all metadata and non-audio columns:
    # You can use the lyrics column if you want, but it requires additional text processing and may not be
    # necessary for good performance.
    # Target is 'genre', features are audio features
    feature_cols = [
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

    X = df[feature_cols]
    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # ENCODING:
    # Use LabelEncoder to encode genre labels numerically.
    # The dataset has 10 distinct genre classes that need to be converted to integers (0-9).
    # This is required for sklearn models which expect numeric target values.
    # TODO: Encode genre labels (use LabelEncoder from sklearn)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # SCALING:
    # Use StandardScaler for LogisticRegression to standardize features (zero mean, unit variance).
    # XGBoost handles feature scaling internally, so do NOT scale features when using XGBoost.
    # This means you may need different scaling strategies per model type:
    # - For LogisticRegression: scale the features before training
    # - For XGBoost: use original (unscaled) features
    # TODO: Scale features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(label_encoder, "models/label_encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    # Get hyperparameters
    train_params = params.get("train", {})

    logger.info(f"Training {len(train_params)} model types...")

    # MODEL TRAINING LOOP STRUCTURE:
    # The model loop should iterate through each model configuration in params['train'].
    # Each model type (logistic_regression, xgboost) has its own hyperparameters and
    # requires different preprocessing. The general structure should follow this pseudocode:
    #
    # for model_name, model_params in train_params.items():
    #     # Determine which model class and features to use
    #     if model_name == 'logistic_regression':
    #         model = LogisticRegression(**model_params)
    #         X_to_use = X_scaled  # Use scaled features
    #     elif model_name == 'xgboost':
    #         model = xgb.XGBClassifier(**model_params)
    #         X_to_use = X  # Use original features (XGBoost handles scaling)
    #
    #     # Start MLflow run to track this model
    #     with mlflow.start_run(run_name=model_name):
    #         # Log all hyperparameters from the config
    #         mlflow.log_params(model_params)
    #
    #         # Train the model
    #         model.fit(X_to_use, y_encoded)
    #
    #         # Evaluate on training data and log metrics
    #         y_pred = model.predict(X_to_use)
    #         accuracy = calculate_accuracy(y, y_pred)
    #         mlflow.log_metric("accuracy", accuracy)
    #
    #         # Save the trained model to MLflow
    #         mlflow.sklearn.log_model(model, artifact_path="model")
    #         # OR for XGBoost: mlflow.xgboost.log_model(model, artifact_path="model")

    # TODO: Loop through each model in train_params and:
    #  1. Create appropriate model instance based on model_name:
    #     - 'logistic_regression': LogisticRegression(**params)
    #     - 'xgboost': xgb.XGBClassifier(**params)
    #  2. Start MLflow run with run_name=model_name
    #  3. Log parameters from config
    #  4. Fit model on features and encoded target
    #     - Use scaled X for LogisticRegression
    #     - Use original X for XGBoost (it handles feature scaling internally)
    #  5. Calculate accuracy metric (and optionally precision, recall, F1)
    #  6. Log metrics to MLflow
    #  7. Log model artifact with appropriate MLflow function
    #  8. End run
    for model_name, model_params in train_params.items():
        logger.info(f"Training model: {model_name}")

        if model_name == "logistic_regression":
            model = LogisticRegression(**model_params)
            X_to_use = X_scaled
            log_model = mlflow.sklearn.log_model

        elif model_name == "xgboost":
            model = xgb.XGBClassifier(**model_params)
            X_to_use = X
            log_model = mlflow.xgboost.log_model

        else:
            logger.warning(f"Skipping unsupported model type: {model_name}")
            continue

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_params(model_params)

            model.fit(X_to_use, y_encoded)

            y_pred = model.predict(X_to_use)
            accuracy = accuracy_score(y_encoded, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            log_model(model, artifact_path="model")

            joblib.dump(model, f"models/{model_name}.joblib")

            logger.info(f"{model_name} accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
