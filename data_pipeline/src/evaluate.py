import mlflow
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_and_register(train_data_path: str = "data/train.csv"):
    """
    Find the best performing model and register it with @champion alias.

    The scaffolding below handles connecting to MLflow and finding the best run.
    Your job is to register that model in the MLflow Model Registry and assign
    the 'champion' alias so the Dockerfile can pull it by name.

    MLflow Model Registry API:
        client.create_model_version(name, source, run_id)
            -> returns a ModelVersion object with a .version attribute
        client.set_registered_model_alias(name, alias, version)
            -> assigns a named alias to a specific version
    """
    logger.info("Evaluating models and registering the best one...")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "spotify-genre-classification"
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    logger.info(f"Searching runs in experiment: {experiment.name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.accuracy >= 0",
        order_by=["metrics.accuracy DESC"],
        max_results=100,
    )

    if not runs:
        raise ValueError("No runs with accuracy found. Did you run train.py?")

    best_run = runs[0]
    best_accuracy = best_run.data.metrics.get("accuracy", 0)
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "spotify-genre-classifier"

    logger.info(f"Best run: {best_run.info.run_id} (accuracy={best_accuracy:.4f})")

    # TODO: Register the model and assign the 'champion' alias
    #   1. Call client.create_model_version() to register model_uri under model_name
    #   2. Call client.set_registered_model_alias() to tag that version as "champion"
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    model_version = client.create_model_version(
        name=model_name, source=model_uri, run_id=best_run.info.run_id
    )

    client.set_registered_model_alias(
        name=model_name, alias="champion", version=model_version.version
    )

    logger.info(
        f"Registered model {model_name} version {model_version.version} "
        "with alias champion"
    )

    metrics = {
        "best_run_id": best_run.info.run_id,
        "best_accuracy": best_accuracy,
        "model_type": best_run.data.params.get("model", "unknown"),
        "model_name": model_name,
        "champion_alias": "champion",
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete. Metrics saved to metrics.json")


if __name__ == "__main__":
    evaluate_and_register()
