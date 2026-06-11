import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

GENRE_LABELS = [
    "Blues",
    "Classical",
    "Country",
    "Electronic",
    "Folk",
    "Hip-Hop",
    "Jazz",
    "Pop",
    "R&B",
    "Rock",
]


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


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the MLflow model baked into the Docker image or local ./models path."""
    model_path = os.getenv("MODEL_PATH", "./models")
    mlmodel_path = Path(model_path) / "MLmodel"

    if not mlmodel_path.exists():
        logger.warning("Model not found at %s. Using fallback response.", model_path)
        return None

    logger.info("Loading MLflow model from %s", model_path)
    return mlflow.pyfunc.load_model(model_path)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming prediction requests as JSONL for drift monitoring."""
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()

        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            if isinstance(payload, dict):
                payload["timestamp"] = datetime.utcnow().isoformat()

                logs_dir = Path("logs")
                logs_dir.mkdir(parents=True, exist_ok=True)

                with open(logs_dir / "api_requests.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")

        except json.JSONDecodeError:
            logger.warning("Could not parse request body as JSON.")

        async def receive():
            return {
                "type": "http.request",
                "body": body_bytes,
                "more_body": False,
            }

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
        return predict_genre(features)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


def _map_prediction_to_genre(raw_prediction: Any) -> str:
    if isinstance(raw_prediction, str):
        if raw_prediction.isdigit():
            index = int(raw_prediction)
            if 0 <= index < len(GENRE_LABELS):
                return GENRE_LABELS[index]
        return raw_prediction

    try:
        index = int(raw_prediction)
        if 0 <= index < len(GENRE_LABELS):
            return GENRE_LABELS[index]
    except (TypeError, ValueError):
        pass

    return str(raw_prediction)


def predict_genre(features: SpotifyFeatures) -> PredictionResponse:
    """Run model inference and return genre plus confidence."""
    model = load_model()

    if model is None:
        return PredictionResponse(genre="Pop", confidence=0.0)

    payload = features.model_dump()
    input_df = pd.DataFrame([[payload[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    prediction = model.predict(input_df)
    genre = _map_prediction_to_genre(prediction[0])

    confidence = 0.0
    predict_fn = getattr(model, "predict_proba", None)
    if callable(predict_fn):
        probabilities = model.predict_proba(input_df)
        confidence = float(probabilities[0].max())

    return PredictionResponse(genre=genre, confidence=confidence)
