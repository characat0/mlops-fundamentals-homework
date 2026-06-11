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
    # You can use the lyrics column if you want, but it requires additional text processing and may not be
    # necessary for good performance.
    # Target is 'genre', features are audio features
        # Feature selection - todas las columnas excepto 'genre', 'year' y metadata
    exclude_cols = ['genre', 'year', 'id', 'name', 'album_name', 'artists', 
                    'lyrics', 'popularity', 'total_artist_followers', 
                    'avg_artist_popularity', 'artist_ids', 'niche_genres']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')
    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Features: {list(X.columns)}")

    # ENCODING:
    # Use LabelEncoder to encode genre labels numerically.
    # The dataset has 10 distinct genre classes that need to be converted to integers (0-9).
    # This is required for sklearn models which expect numeric target values.

    
    # TODO: Encode genre labels (use LabelEncoder from sklearn)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # SCALING:
    # Use StandardScaler for LogisticRegression to standardize features (zero mean, unit variance).
    # XGBoost handles feature scaling internally, so do NOT scale features when using XGBoost.
    # This means you may need different scaling strategies per model type:
    # - For LogisticRegression: scale the features before training
    # - For XGBoost: use original (unscaled) features
    # TODO: Scale features using StandardScaler
    joblib.dump(le, 'label_encoder.pkl')
    logger.info("LabelEncoder saved to label_encoder.pkl")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    logger.info("StandardScaler saved to scaler.pkl")

    # Get hyperparameters
    train_params = params.get("train", {})

    logger.info(f"Training {len(train_params)} model types...")

    
    for model_name, model_params in train_params.items():
        logger.info(f"Training {model_name} with params: {model_params}")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(model_params)
            
            # Select model and features
            if model_name == 'logistic_regression':
                model = LogisticRegression(**model_params)
                X_to_use = X_scaled
                logger.info("Using scaled features for Logistic Regression")
            elif model_name == 'xgboost':
                model = xgb.XGBClassifier(**model_params)
                X_to_use = X
                logger.info("Using original features for XGBoost")
            else:
                continue
            
            # Train model
            model.fit(X_to_use, y_encoded)
            
            # Evaluate
            y_pred = model.predict(X_to_use)
            accuracy = (y_pred == y_encoded).mean()
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            if model_name == 'logistic_regression':
                mlflow.sklearn.log_model(model, artifact_path="model")
            else:
                mlflow.xgboost.log_model(model, artifact_path="model")
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
