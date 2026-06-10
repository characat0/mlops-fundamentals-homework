import json
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EXPERIMENT_NAME = "spotify-genre-classifier"
MODEL_NAME = "spotify-genre-classifier"
CHAMPION_ALIAS = "champion"


def _compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _evaluate_on_data(
    model, df: pd.DataFrame, label_encoder, scaler, feature_order
) -> dict:
    """Score a champion on a held-out DataFrame, mirroring the training
    preprocessing so the comparison is fair."""
    X = df[feature_order].copy()
    y_raw = df["genre"]
    impl = getattr(model, "_model_impl", model)
    underlying = getattr(impl, "xgb_model", impl)
    is_tree_based = underlying.__class__.__name__ == "XGBClassifier"
    X_input = X if is_tree_based else scaler.transform(X)
    y_encoded_true = label_encoder.transform(y_raw)
    y_pred = model.predict(X_input)
    return _compute_metrics(y_encoded_true, y_pred), len(df)


def evaluate_and_register(
    train_data_path: str = "data/train.csv",
    prod_data_path: str = "data/prod_sim.csv",
    models_dir: str = "models",
) -> None:
    """
    Find the best performing model, score it on the production-simulation
    split (years >2010) to quantify temporal drift, and register the
    champion with @champion alias.
    """
    logger.info("Evaluating models and registering the best one...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.error(f"Experiment '{EXPERIMENT_NAME}' not found. Did you run train.py?")
        return
    logger.info(f"Searching runs in experiment: {experiment.name}")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=100,
    )
    finished_runs = [r for r in runs if r.info.status == "FINISHED"]
    if not finished_runs:
        logger.error("No FINISHED runs found. Did you run train.py?")
        return
    best_run = finished_runs[0]
    test_accuracy = best_run.data.metrics.get("accuracy", 0)
    train_accuracy = best_run.data.metrics.get("train_accuracy", 0)
    model_uri = f"runs:/{best_run.info.run_id}/model"
    logger.info(
        f"Best run: {best_run.info.run_id} "
        f"(test_accuracy={test_accuracy:.4f}, train_accuracy={train_accuracy:.4f}, "
        f"model_type={best_run.data.tags.get('model_type', 'unknown')})"
    )
    models_path = Path(models_dir)
    label_encoder = joblib.load(models_path / "label_encoder.joblib")
    scaler = joblib.load(models_path / "scaler.joblib")
    feature_order = joblib.load(models_path / "feature_order.joblib")
    champion = mlflow.pyfunc.load_model(model_uri)
    prod_metrics, prod_n = _evaluate_on_data(
        champion,
        pd.read_csv(prod_data_path),
        label_encoder,
        scaler,
        feature_order,
    )
    train_metrics_eval, train_n = _evaluate_on_data(
        champion,
        pd.read_csv(train_data_path),
        label_encoder,
        scaler,
        feature_order,
    )
    drift_pct = (
        100.0
        * (train_metrics_eval["accuracy"] - prod_metrics["accuracy"])
        / train_metrics_eval["accuracy"]
        if train_metrics_eval["accuracy"] > 0
        else 0.0
    )
    logger.info(
        f"Champion on train split (n={train_n}): "
        f"accuracy={train_metrics_eval['accuracy']:.4f}, f1={train_metrics_eval['f1_macro']:.4f}"
    )
    logger.info(
        f"Champion on prod_sim (n={prod_n}, years>2010): "
        f"accuracy={prod_metrics['accuracy']:.4f}, f1={prod_metrics['f1_macro']:.4f}"
    )
    logger.info(
        f"Temporal accuracy drift: {drift_pct:+.2f}% "
        f"({train_metrics_eval['accuracy']:.4f} -> {prod_metrics['accuracy']:.4f})"
    )
    try:
        client.create_registered_model(MODEL_NAME)
        logger.info(f"Created registered model '{MODEL_NAME}'")
    except mlflow.exceptions.RestException as exc:
        if "RESOURCE_ALREADY_EXISTS" in str(exc):
            logger.info(f"Registered model '{MODEL_NAME}' already exists")
        else:
            raise
    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=best_run.info.run_id,
    )
    logger.info(
        f"Registered model version {model_version.version} "
        f"for run {best_run.info.run_id}"
    )
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=model_version.version,
    )
    logger.info(
        f"Assigned alias '@{CHAMPION_ALIAS}' to version {model_version.version}"
    )
    metrics = {
        "best_run_id": best_run.info.run_id,
        "model_type": best_run.data.tags.get("model_type", "unknown"),
        "model_name": MODEL_NAME,
        "model_version": model_version.version,
        "champion_alias": CHAMPION_ALIAS,
        "train_split": {
            "n_samples": train_n,
            "accuracy": train_metrics_eval["accuracy"],
            "precision_macro": train_metrics_eval["precision_macro"],
            "recall_macro": train_metrics_eval["recall_macro"],
            "f1_macro": train_metrics_eval["f1_macro"],
        },
        "test_split": {
            "accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
        },
        "prod_sim": {
            "n_samples": prod_n,
            "accuracy": prod_metrics["accuracy"],
            "precision_macro": prod_metrics["precision_macro"],
            "recall_macro": prod_metrics["recall_macro"],
            "f1_macro": prod_metrics["f1_macro"],
            "temporal_drift_pct": drift_pct,
        },
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation complete. Metrics saved to metrics.json")
    logger.info(
        f"Champion: {MODEL_NAME} v{model_version.version} "
        f"(@{CHAMPION_ALIAS}, train={train_metrics_eval['accuracy']:.4f} "
        f"-> prod_sim={prod_metrics['accuracy']:.4f}, drift={drift_pct:+.2f}%)"
    )


if __name__ == "__main__":
    evaluate_and_register()
