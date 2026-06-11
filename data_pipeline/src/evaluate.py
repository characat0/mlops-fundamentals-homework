import argparse
import json
import logging
import os

import mlflow
from mlflow.exceptions import MlflowException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_all_runs(client):
    experiments = client.search_experiments()
    experiment_ids = [experiment.experiment_id for experiment in experiments]

    if not experiment_ids:
        return []

    return client.search_runs(
        experiment_ids=experiment_ids,
        filter_string="metrics.accuracy IS NOT NULL",
        order_by=["metrics.accuracy DESC"],
        max_results=100
    )


def evaluate_and_register(train_data_path: str = "data/train.csv"):
    """
    Find the best performing model and register it with the @champion alias.

    Args:
        train_data_path: Kept as a pipeline dependency hook for DVC.
    """
    logger.info("Evaluating models and registering the best one...")
    logger.info(f"Using training data dependency: {train_data_path}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    runs = _get_all_runs(client)

    if not runs:
        logger.error("No runs found. Did you run train.py?")
        return

    best_run = runs[0]
    best_accuracy = best_run.data.metrics.get("accuracy", 0)
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "spotify-genre-classifier"

    logger.info(f"Best run: {best_run.info.run_id} (accuracy={best_accuracy:.4f})")

    try:
        client.create_registered_model(model_name)
    except MlflowException:
        logger.info(f"Registered model already exists: {model_name}")

    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run.info.run_id
    )
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version.version
    )

    metrics = {
        "best_run_id": best_run.info.run_id,
        "best_accuracy": best_accuracy,
        "model_type": best_run.data.params.get("model", "unknown"),
        "model_name": model_name,
        "model_version": model_version.version,
        "champion_alias": "champion"
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete. Metrics saved to metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/train.csv")
    args = parser.parse_args()

    evaluate_and_register(args.train_data)
