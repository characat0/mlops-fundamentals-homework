import mlflow
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_and_register(train_data_path: str = "data/train.csv"):
    """
    Find the best performing model and register it with @champion alias.

    Steps:
    1. Query MLflow API to find all runs
    2. Compare by accuracy metric
    3. Register the best model in MLflow Model Registry
    4. Assign alias 'champion'
    5. Save metrics summary to metrics.json
    """
    logger.info("Evaluating models and registering the best one...")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    # Usar experimento Default
    experiment = client.get_experiment_by_name("Default") or client.get_experiment("0")
    logger.info(f"Searching runs in experiment: {experiment.name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=100
    )

    if not runs:
        logger.error("No runs found. Did you run train.py?")
        return

    # Seleccionar el mejor run por accuracy
    best_run = runs[0]
    best_accuracy = best_run.data.metrics.get("accuracy", 0)
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "spotify-genre-classifier"

    logger.info(f"Best run: {best_run.info.run_id} (accuracy={best_accuracy:.4f})")

    # Crear el modelo registrado si no existe
    try:
        client.create_registered_model(model_name)
        logger.info(f"Registered model '{model_name}' created.")
    except Exception as e:
        logger.info(f"Model '{model_name}' already exists or could not be created: {e}")

    # Crear nueva versión del modelo
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run.info.run_id
    )

    # Asignar alias champion
    client.set_registered_model_alias(model_name, "champion", model_version.version)

    # Guardar métricas en metrics.json
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
    logger.info("Best model registered as @champion in MLflow Model Registry")


if __name__ == "__main__":
    evaluate_and_register()
