import argparse
import json
import logging
import os
import time

import mlflow
from mlflow.exceptions import MlflowException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_registered_model(client, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        logger.info("Creating registered model: %s", model_name)
        client.create_registered_model(model_name)


def evaluate_and_register(train_data_path: str = "data/train.csv") -> None:
    """Find the best MLflow run and register it as champion."""
    del train_data_path

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    logger.info("Using MLflow tracking URI: %s", tracking_uri)
    logger.info("Searching MLflow runs by accuracy")

    experiments = client.search_experiments()
    experiment_ids = [experiment.experiment_id for experiment in experiments]

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string="metrics.accuracy >= 0",
        order_by=["metrics.accuracy DESC"],
        max_results=100,
    )

    if not runs:
        raise RuntimeError("No MLflow runs found. Run dvc repro/train.py first.")

    best_run = runs[0]
    best_accuracy = best_run.data.metrics.get("accuracy", 0.0)
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "spotify-genre-classifier"

    logger.info("Best run: %s accuracy=%.4f", best_run.info.run_id, best_accuracy)

    _ensure_registered_model(client, model_name)

    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run.info.run_id,
    )

    for _ in range(30):
        current_version = client.get_model_version(
            name=model_name,
            version=model_version.version,
        )
        if current_version.status == "READY":
            break
        time.sleep(1)

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version.version,
    )

    metrics = {
        "best_run_id": best_run.info.run_id,
        "best_accuracy": best_accuracy,
        "model_type": best_run.data.params.get("model", "unknown"),
        "model_name": model_name,
        "model_version": model_version.version,
        "champion_alias": "champion",
    }

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Registered %s version %s as @champion", model_name, model_version.version)
    logger.info("Evaluation complete. Metrics saved to metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/train.csv")
    args = parser.parse_args()

    evaluate_and_register(train_data_path=args.train_data)
