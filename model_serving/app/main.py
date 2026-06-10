from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
import joblib
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms"
]

MODEL_DIR = Path("./models")
_MODEL = None
_LABEL_ENCODER = None


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
    """Log incoming /predict requests to logs/api_requests.jsonl."""
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()

        try:
            payload = json.loads(body_bytes.decode("utf-8"))
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()

            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            with (logs_dir / "api_requests.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except json.JSONDecodeError:
            logger.warning("Could not parse /predict request body as JSON")

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


def _load_model():
    global _MODEL, _LABEL_ENCODER

    if _MODEL is not None:
        return _MODEL, _LABEL_ENCODER

    if not MODEL_DIR.exists():
        return None, None

    _MODEL = mlflow.sklearn.load_model(str(MODEL_DIR))

    label_encoder_path = MODEL_DIR / "label_encoder.joblib"
    if label_encoder_path.exists():
        _LABEL_ENCODER = joblib.load(label_encoder_path)

    return _MODEL, _LABEL_ENCODER


def _features_to_frame(features: SpotifyFeatures) -> pd.DataFrame:
    values = {name: getattr(features, name) for name in FEATURE_NAMES}
    return pd.DataFrame([values], columns=FEATURE_NAMES)


def _decode_genre(raw_prediction, label_encoder) -> str:
    prediction = raw_prediction[0]

    if isinstance(prediction, str):
        return prediction

    if label_encoder is not None:
        return str(label_encoder.inverse_transform([int(prediction)])[0])

    fallback_genres = [
        "Blues", "Classical", "Country", "Electronic", "Folk",
        "Hip-Hop", "Jazz", "Pop", "R&B", "Rock"
    ]
    return fallback_genres[int(prediction) % len(fallback_genres)]


def predict_genre(features: SpotifyFeatures) -> PredictionResponse:
    """Predict a genre using the baked MLflow model when it is available."""
    model, label_encoder = _load_model()

    if model is None:
        logger.warning("No local model found at ./models; returning fallback prediction")
        return PredictionResponse(genre="Pop", confidence=0.0)

    feature_frame = _features_to_frame(features)
    prediction = model.predict(feature_frame)
    genre = _decode_genre(prediction, label_encoder)

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)
        confidence = float(probabilities[0].max())

    return PredictionResponse(genre=genre, confidence=confidence)
