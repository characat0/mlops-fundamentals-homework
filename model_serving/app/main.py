from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
import mlflow
import pandas as pd
import shutil

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

_model = None
_label_encoder = None

FEATURE_NAMES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms', 'popularity',
    'total_artist_followers', 'avg_artist_popularity'
]

GENRE_LABELS = [
    'Blues', 'Classical', 'Country', 'Electronic', 'Folk',
    'Hip-Hop', 'Jazz', 'Pop', 'R&B', 'Rock'
]

# TODO: Define the SpotifyFeatures Pydantic model.
#
# Include the audio feature fields from the Kaggle dataset, with the correct
# Python types. The field names must match the column names exactly
# (the tests send a payload with these exact keys).
#
# Example fields and types:
#   danceability (float), energy (float), key (int), loudness (float),
#   mode (int), speechiness (float), acousticness (float),
#   instrumentalness (float), liveness (float), valence (float),
#   tempo (float), duration_ms (int)
class SpotifyFeatures(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    popularity: float = 0.0
    total_artist_followers: float = 0.0
    avg_artist_popularity: float = 0.0


class PredictionResponse(BaseModel):
    genre: str
    confidence: float = 0.0

def _is_valid_model_dir(path: Path) -> bool:
    """Verifica que el directorio contenga un modelo MLflow válido."""
    return any(path.rglob("MLmodel"))


def load_model():
    global _model

    if _model is not None:
        return _model

    # 1. Modelo bakeado en Docker
    docker_path = Path("./models")
    if docker_path.exists() and _is_valid_model_dir(docker_path):
        try:
            _model = _load_model_auto_flavor(docker_path.resolve().as_uri())
            logger.info("Modelo cargado desde ./models/")
            return _model
        except Exception as e:
            logger.warning(f"No se pudo cargar desde ./models/: {e}")

    # 2. Descargar desde MLflow Registry a ruta local
    cache_dir = Path("./models_cache")
    cache_valid = cache_dir.exists() and _is_valid_model_dir(cache_dir)

    if not cache_valid:
        if cache_dir.exists():
            logger.warning("Cache inválido o corrupto, eliminando y re-descargando...")
            shutil.rmtree(cache_dir)
        logger.info("Descargando modelo desde MLflow Registry...")
        cache_dir.mkdir(exist_ok=True)
        mlflow.artifacts.download_artifacts(
            artifact_uri="models:/genre-classifier@champion",
            dst_path=str(cache_dir.resolve()),
        )

    # Buscar el subdirectorio que contiene MLmodel
    model_dir = cache_dir
    for candidate in cache_dir.rglob("MLmodel"):
        model_dir = candidate.parent
        break

    try:
        model_uri = model_dir.resolve().as_uri()
        _model = _load_model_auto_flavor(model_uri)
        logger.info(f"Modelo cargado desde {model_uri}")
        return _model
    except Exception as e:
        logger.error(f"Fallo al cargar modelo desde cache: {e}. Limpiando cache...")
        shutil.rmtree(cache_dir)
        raise RuntimeError(
            f"Modelo en cache inválido y fue eliminado. Reinicia el servicio para re-descargarlo. Error: {e}"
        ) from e


def _load_model_auto_flavor(model_uri: str):
    """Detecta el flavor del modelo leyendo MLmodel y carga con el loader correcto."""
    import yaml

    # Convertir URI a path para leer MLmodel
    if model_uri.startswith("file:///"):
        mlmodel_path = Path(model_uri[8:]) / "MLmodel"  # quitar file:///
    else:
        mlmodel_path = Path(model_uri) / "MLmodel"

    with open(mlmodel_path, "r") as f:
        mlmodel = yaml.safe_load(f)

    flavors = mlmodel.get("flavors", {})
    logger.info(f"Flavors disponibles: {list(flavors.keys())}")

    if "sklearn" in flavors:
        return mlflow.sklearn.load_model(model_uri)
    elif "xgboost" in flavors:
        import mlflow.xgboost
        return mlflow.xgboost.load_model(model_uri)
    elif "python_function" in flavors:
        return mlflow.pyfunc.load_model(model_uri)
    else:
        raise RuntimeError(f"Flavor no soportado. Disponibles: {list(flavors.keys())}")

def load_label_encoder():
    global _label_encoder
    if _label_encoder is not None:
        return _label_encoder
 
    # Buscar label_encoder junto al modelo en models_cache
    for candidate in Path("./models_cache").rglob("label_encoder.pkl"):
        with open(candidate, "rb") as f:
            _label_encoder = pickle.load(f)
        return _label_encoder
 
    # Fallback: ruta relativa al repo (desarrollo local)
    fallback = (
        Path(__file__).resolve().parents[2]
        / "data_pipeline" / "src" / "y_encoded.pkl"
    )
    if fallback.exists():
        with open(fallback, "rb") as f:
            _label_encoder = pickle.load(f)
        return _label_encoder
 
    return None  # Usará GENRE_LABELS hardcodeado

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming /predict requests to logs/api_requests.jsonl.

    Logging here (middleware) rather than inside the endpoint keeps
    observability separate from business logic — easier to disable, test,
    and extend (rate limiting, metrics) without touching endpoint code.

    TODO:
      1. Only log POST requests to "/predict"
      2. Read the body: body_bytes = await request.body()
      3. Parse as JSON, add a "timestamp" field (datetime.utcnow().isoformat())
      4. Append a JSON line to logs/api_requests.jsonl (create logs/ if needed)
      5. Reconstruct the request so the endpoint can still read it:
             async def receive():
                 return {"type": "http.request", "body": body_bytes}
             request = Request(request.scope, receive)
      6. Call response = await call_next(request) and return it
    """
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()


        payload = json.loads(body_bytes)

        payload["timestamp"] = datetime.utcnow().isoformat()

        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        with open(logs_dir / "api_requests.jsonl", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload) + "\n")

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request = Request(request.scope, receive)

    response = await call_next(request)
    return response


# TODO: Implement the GET /health endpoint.
#   It should return {"status": "healthy"} with a 200 status code.
#   This is used by load balancers and CI checks to verify the API is up.
@app.get("/health")
def health() -> dict:
    """Health check endpoint used by load balancers and CI."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: SpotifyFeatures) -> PredictionResponse:
    """Predict Spotify track genre from audio features."""
    try:
        prediction = predict_genre(features)
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


def predict_genre(features: SpotifyFeatures) -> PredictionResponse:
    """
    **IMPORTANT: This is an intentionally incomplete skeleton for students to implement.**

    Students must:
    1. Load the MLflow model registered with the @champion alias
       - The model is baked into the Docker container at ./models/
       - Use: mlflow.sklearn.load_model("./models")
    2. Convert SpotifyFeatures to the format expected by the model
       - Extract feature values in the correct order (order matters for sklearn models)
       - Must match the audio features used during training
    3. Perform inference on the audio features
    4. Map the predicted class index back to genre name
    5. Return a PredictionResponse with the genre and confidence score

    Example implementation structure:
        import mlflow

        model = mlflow.sklearn.load_model("./models")

        feature_names = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        feature_vector = [getattr(features, name) for name in feature_names]

        prediction = model.predict([feature_vector])
        probabilities = model.predict_proba([feature_vector])
        confidence = float(probabilities[0].max())

        # Map numeric class index back to genre label using the LabelEncoder
        # you saved during training, or hardcode the genre list if consistent.

        return PredictionResponse(genre=predicted_genre, confidence=confidence)

    For now, returns a placeholder so API tests pass:
    """
    model = load_model()
    feature_vector = [getattr(features, name) for name in FEATURE_NAMES]

    logger.info(f"feature_vector length: {len(feature_vector)}")
    logger.info(f"FEATURE_NAMES length: {len(FEATURE_NAMES)}")
    logger.info(f"FEATURE_NAMES: {FEATURE_NAMES}")

    df = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)

    if hasattr(model, "predict_proba"):
        prediction = model.predict(df)
        probabilities = model.predict_proba(df)
        confidence = float(probabilities[0].max())
    else:
        result = model.predict(df)
        prediction = result.values if hasattr(result, "values") else result
        confidence = 0.0  # pyfunc no expone probabilidades directamente

    label_encoder = load_label_encoder()
    if label_encoder is not None:
        predicted_genre = label_encoder.inverse_transform(
            [int(prediction[0])]
        )[0]
    else:
        predicted_genre = GENRE_LABELS[int(prediction[0])]

    return PredictionResponse(genre=predicted_genre, confidence=confidence)
    # return PredictionResponse(genre="Pop", confidence=0.85)
