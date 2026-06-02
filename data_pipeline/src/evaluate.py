import mlflow
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_and_register(train_data_path: str = "data/train.csv"):
    """
    Find the best performing model and register it with @champion alias.

    TODO:
        1. Connect to MLflow (assumes server is running at MLFLOW_TRACKING_URI)
        2. Query all runs from the current experiment
        3. Find the run with the highest accuracy metric
        4. Register the best model in MLflow Model Registry
        5. Assign the alias '@champion' to this registered model version
        6. Save evaluation metrics to metrics.json

    Example code for registering model:
        # model_uri = f"runs:/{best_run.info.run_id}/model"
        # registered_model = client.create_model_version(
        #     name=model_name,
        #     source=model_uri,
        #     run_id=best_run.info.run_id
        # )
        # client.set_registered_model_alias(
        #     name=model_name,
        #     alias="champion",
        #     version=registered_model.version
        # )
    """
    logger.info("Evaluating models and registering the best one...")

    # Set MLflow tracking URI (from environment or default)
    import os
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    # Get the current experiment (assumes tracking in default experiment)
    experiment = client.get_experiment_by_name(None) or client.get_experiment("0")

    logger.info(f"Searching runs in experiment: {experiment.name}")

    # Find the best run by accuracy
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

    logger.info(f"Best run: {best_run.info.run_id}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Model type: {best_run.data.params.get('model', 'unknown')}")

    # Register the model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "spotify-genre-classifier"

    # TODO: Register the model in MLflow Model Registry
    # TODO: Assign the @champion alias to this model version

    # Save metrics to metrics.json
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
