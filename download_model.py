import mlflow

# Apuntar al servidor MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# URI del modelo champion
model_uri = "models:/spotify-genre-classifier@champion"

# Descargar el modelo a ./models
mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path="./models"
)



