import mlflow
import json
import logging

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

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(
        "spotify-genre-classifier"
    )

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
        client.create_registered_model(
            model_name
        )
    except Exception:
        pass

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
        "champion_alias": "champion"
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete. Metrics saved to metrics.json")


if __name__ == "__main__":
    evaluate_and_register()
