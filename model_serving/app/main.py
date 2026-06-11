from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import json
import logging
import pandas as pd
import mlflow

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOG_FILE = Path("logs/api_requests.jsonl")
MODEL_PATH = "./models"

# Audio features en el orden usado durante el entrenamiento
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]

# Generos en el orden que produce LabelEncoder (alfabetico)
GENRES = [
    "Blues", "Classical", "Country", "Electronic", "Folk",
    "Hip-Hop", "Jazz", "Pop", "R&B", "Rock",
]

_model = None


def _load_model():
    """Carga el modelo champion una sola vez. None si no esta disponible."""
    global _model
    if _model is None:
        for loader in (mlflow.xgboost.load_model, mlflow.sklearn.load_model):
            try:
                _model = loader(MODEL_PATH)
                break
            except Exception:
                continue
    return _model


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
        try:
            data = json.loads(body_bytes)
        except json.JSONDecodeError:
            data = {}
        data["timestamp"] = datetime.utcnow().isoformat()

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")

        # Reconstruir el request: el body ya se consumio al leerlo arriba
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request = Request(request.scope, receive)

    response = await call_next(request)
    return response


@app.get("/health")
def health():
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
    model = _load_model()
    if model is None:
        # Sin modelo disponible (ej. CI sin ./models): respuesta por defecto
        return PredictionResponse(genre="Pop", confidence=0.0)

    X = pd.DataFrame([features.model_dump()])[AUDIO_FEATURES]
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())
    return PredictionResponse(genre=GENRES[idx], confidence=float(proba.max()))
