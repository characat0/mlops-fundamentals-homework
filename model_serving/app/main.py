from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import logging
import os
from pathlib import Path
from datetime import datetime

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()

        try:
            body_json = json.loads(body_bytes)
            body_json["timestamp"] = datetime.utcnow().isoformat()

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / "api_requests.jsonl"

            with open(log_path, "a") as f:
                f.write(json.dumps(body_json) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log request: {e}")

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
    import mlflow.sklearn
    import numpy as np

    feature_names = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]

    model = mlflow.sklearn.load_model("./models")

    feature_vector = [getattr(features, name) for name in feature_names]

    prediction = model.predict([feature_vector])
    probabilities = model.predict_proba([feature_vector])
    confidence = float(probabilities[0].max())

    try:
        import joblib
        le = joblib.load("./models/label_encoder.pkl")
        predicted_genre = le.inverse_transform(prediction)[0]
    except Exception:
        genres = ['Blues', 'Classical', 'Country', 'Electronic',
                  'Folk', 'Hip-Hop', 'Jazz', 'Pop', 'R&B', 'Rock']
        idx = int(prediction[0])
        predicted_genre = genres[idx] if idx < len(genres) else "Unknown"

    return PredictionResponse(genre=predicted_genre, confidence=confidence)
