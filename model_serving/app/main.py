from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime
import json
import logging
import os
from pathlib import Path

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where the @champion model is baked at Docker build time.
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

# Audio features in the exact order used during training. The order matters:
# sklearn/XGBoost models are positional.
FEATURE_NAMES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]

# Genre labels in the same order LabelEncoder produces (sorted alphabetically),
# used to map a predicted class index back to a genre name.
GENRE_LABELS = sorted([
    "Blues", "Classical", "Country", "Electronic", "Folk",
    "Hip-Hop", "Jazz", "Pop", "R&B", "Rock",
])

# Lazily loaded singleton model (loaded once on first prediction).
_model = None
_model_loaded = False


class SpotifyFeatures(BaseModel):
    """Audio features of a single Spotify track (the /predict payload)."""

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


class PredictionResponse(BaseModel):
    genre: str
    confidence: float = 0.0


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming /predict requests to logs/api_requests.jsonl.

    Logging here (middleware) rather than inside the endpoint keeps
    observability separate from business logic — easier to disable, test,
    and extend (rate limiting, metrics) without touching endpoint code.

    Only POST /predict requests are logged. Each line is the JSON payload plus
    a UTC timestamp; these logs feed the online drift analysis. The request is
    reconstructed after reading the body so the endpoint can still parse it.
    """
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()

        try:
            payload = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            payload = {}

        payload["timestamp"] = datetime.utcnow().isoformat()

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "api_requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        # Rebuild the request so the endpoint can still read the body.
        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request = Request(request.scope, receive)

    response = await call_next(request)
    return response


@app.get("/health")
def health():
    """Liveness probe used by load balancers and CI."""
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
    """Run inference with the @champion model and return the predicted genre.

    Loads the model (baked into ./models at Docker build time), builds the
    feature vector in the training order, predicts the class index, and maps it
    back to a genre name (the index follows LabelEncoder's alphabetical order).

    If the model artifact is unavailable (e.g. CI runs with no baked model),
    falls back to a safe placeholder so the API stays up and tests pass.
    """
    model = _get_model()

    feature_vector = [getattr(features, name) for name in FEATURE_NAMES]

    if model is None:
        # No model baked in (CI / local dev without `mlflow models download`).
        logger.warning("No model loaded; returning placeholder prediction.")
        return PredictionResponse(genre="Pop", confidence=0.0)

    prediction = model.predict([feature_vector])
    predicted_index = int(prediction[0])

    confidence = 0.0
    try:
        probabilities = model.predict_proba([feature_vector])
        confidence = float(probabilities[0].max())
    except Exception:  # model may not expose predict_proba
        confidence = 0.0

    if 0 <= predicted_index < len(GENRE_LABELS):
        genre = GENRE_LABELS[predicted_index]
    else:
        genre = str(predicted_index)

    return PredictionResponse(genre=genre, confidence=confidence)


def _resolve_model_dir():
    """Find the directory that actually contains the MLmodel file.

    `mlflow artifacts download -d ./models` can nest the artifact under
    ./models/model (or another subfolder), so check the common layouts first
    and then fall back to a recursive search for the MLmodel file.
    """
    for candidate in (MODEL_PATH, os.path.join(MODEL_PATH, "model")):
        if os.path.exists(os.path.join(candidate, "MLmodel")):
            return candidate
    for root, _dirs, files in os.walk(MODEL_PATH):
        if "MLmodel" in files:
            return root
    return MODEL_PATH


def _get_model():
    """Load the @champion model from ./models once and cache it.

    Loads with the loader that matches the model's flavor (the champion may be
    either sklearn LogisticRegression or XGBoost). Returns None if the model
    can't be loaded (no artifact, mlflow missing), letting the API degrade
    gracefully instead of crashing.
    """
    global _model, _model_loaded
    if _model_loaded:
        return _model

    _model_loaded = True
    try:
        from mlflow.models import Model

        model_dir = _resolve_model_dir()
        flavors = Model.load(os.path.join(model_dir, "MLmodel")).flavors

        if "xgboost" in flavors:
            import mlflow.xgboost

            _model = mlflow.xgboost.load_model(model_dir)
        else:
            import mlflow.sklearn

            _model = mlflow.sklearn.load_model(model_dir)
        logger.info(f"Loaded champion model from {model_dir} (flavors={list(flavors)})")
    except Exception as exc:  # degrade gracefully on any failure
        logger.warning(f"Could not load model from {MODEL_PATH}: {exc}")
        _model = None
    return _model
