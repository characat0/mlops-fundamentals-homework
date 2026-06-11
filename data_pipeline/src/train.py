import argparse
import mlflow
import yaml
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score

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

    # FEATURE SELECTION:
    # Students should select features from the Kaggle dataset.
    # Drop all metadata and non-audio columns:
    # You can use the lyrics column if you want,
    # but it requires additional text processing and may not be
    # necessary for good performance.
    # Target is 'genre', features are audio features
    FEATURES = [
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

    X = df[FEATURES]
    y = df["genre"]

    logger.info(
        f"Features shape: {X.shape}, "
        f"Target shape: {y.shape}"
    )

    # ENCODING:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # SCALING:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # MLFLOW
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "spotify-genre-classifier"
    )

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

    for model_name, model_params in train_params.items():

        logger.info(f"Training {model_name}")

        if model_name == "logistic_regression":

            model = LogisticRegression(
                **model_params
            )

            X_to_use = X_scaled

        elif model_name == "xgboost":

            model = xgb.XGBClassifier(
                **model_params,
                objective="multi:softprob",
                eval_metric="mlogloss"
            )

            X_to_use = X

        else:
            logger.warning(f"Unknown model: {model_name}")
            continue

        with mlflow.start_run(run_name=model_name):

            mlflow.log_params(model_params)

            model.fit(X_to_use, y_encoded)

            y_pred = model.predict(X_to_use)

            accuracy = accuracy_score(
                y_encoded,
                y_pred
            )

            mlflow.log_metric(
                "accuracy",
                float(accuracy)
            )

            if model_name == "xgboost":

                mlflow.xgboost.log_model(
                    model,
                    artifact_path="model"
                )

            else:

                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model"
                )

            logger.info(
                f"{model_name} accuracy={accuracy:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
