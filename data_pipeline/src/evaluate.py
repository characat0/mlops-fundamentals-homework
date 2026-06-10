import json
import logging
import os
import sys

import mlflow
from mlflow import MlflowClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EXPERIMENT_NAME = "spotify-genre-classifier"
MODEL_NAME = "spotify-genre-classifier"
CHAMPION_ALIAS = "champion"


def evaluate_and_register(train_data_path: str = "data/train.csv") -> None:
    """
    Find the best performing model and register it with @champion alias.
    """
    logger.info("Evaluating models and registering the best one...")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.error(
            f"Experiment '{EXPERIMENT_NAME}' not found. Did you run train.py?"
        )
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
    best_accuracy = best_run.data.metrics.get("accuracy", 0)
    model_uri = f"runs:/{best_run.info.run_id}/model"

    logger.info(
        f"Best run: {best_run.info.run_id} "
        f"(accuracy={best_accuracy:.4f}, "
        f"model_type={best_run.data.tags.get('model_type', 'unknown')})"
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
        "best_accuracy": best_accuracy,
        "model_type": best_run.data.tags.get("model_type", "unknown"),
        "model_name": MODEL_NAME,
        "model_version": model_version.version,
        "champion_alias": CHAMPION_ALIAS,
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete. Metrics saved to metrics.json")
    logger.info(
        f"Champion: {MODEL_NAME} v{model_version.version} "
        f"(@{CHAMPION_ALIAS}, accuracy={best_accuracy:.4f})"
    )


if __name__ == "__main__":
    evaluate_and_register()
