import argparse
import mlflow
import yaml
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]


def train(data_path: str, params: dict):
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    logger.info(f"Using features: {available_features}")

    X = df[available_features]
    y = df["genre"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info(f"Classes: {list(le.classes_)}")

    joblib.dump(le, "label_encoder.pkl")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_params = params.get("train", {})
    logger.info(f"Training {len(train_params)} model types...")

    for model_name, model_params in train_params.items():
        logger.info(f"Training model: {model_name}")

        if model_name == 'logistic_regression':
            model = LogisticRegression(**model_params)
            X_to_use = X_scaled
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(
                **model_params,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            X_to_use = X.values
        else:
            logger.warning(f"Unknown model type: {model_name}, skipping.")
            continue

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(model_params)
            mlflow.log_param("model", model_name)

            model.fit(X_to_use, y_encoded)

            y_pred = model.predict(X_to_use)
            accuracy = accuracy_score(y_encoded, y_pred)
            precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            logger.info(f"{model_name} — accuracy={accuracy:.4f}, f1={f1:.4f}")

            if model_name == 'xgboost':
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            mlflow.log_artifact("label_encoder.pkl")

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, default="params.yaml")
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    train(args.data_path, params)
