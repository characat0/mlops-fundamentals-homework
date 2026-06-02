import argparse
import mlflow
import yaml
import pandas as pd
import logging
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

    TODO:
        1. Load the training data from data_path
        2. Separate features (X) from target (y)
           - Target: 'genre' column (10 classes)
           - Features: audio features (12 total)
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

    # Target is 'genre', features are audio features
    X = df.drop(["genre", "year"], axis=1, errors='ignore')
    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # TODO: Encode genre labels (use LabelEncoder from sklearn)
    # TODO: Scale features using StandardScaler

    # Get hyperparameters
    train_params = params.get("train", {})

    logger.info(f"Training {len(train_params)} model types...")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
