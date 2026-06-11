import mlflow
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_and_register(train_data_path: str = "data/train.csv"):
    logger.info("Evaluating models and registering the best one...")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(None) or client.get_experiment("0")
    logger.info(f"Searching runs in experiment: {experiment.name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=100
    )

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
        logger.info(f"Created registered model: {model_name}")
    except Exception:
        logger.info(f"Model '{model_name}' already exists in registry.")

    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run.info.run_id
    )
    logger.info(f"Registered model version: {model_version.version}")

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version.version
    )
    logger.info(f"Set alias '@champion' -> version {model_version.version}")

    metrics = {
        "best_run_id": best_run.info.run_id,
        "best_accuracy": best_accuracy,
        "model_type": best_run.data.params.get("model", "unknown"),
        "model_name": model_name,
        "champion_alias": "champion"
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete. Metrics saved to metrics.json")


if __name__ == "__main__":
    evaluate_and_register()
