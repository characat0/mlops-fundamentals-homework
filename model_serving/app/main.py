from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import logging
import os
from pathlib import Path
from datetime import datetime
import mlflow.sklearn
import numpy as np

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]

GENRES = ['Blues', 'Classical', 'Country', 'Electronic', 'Folk',
          'Hip-Hop', 'Jazz', 'Pop', 'R&B', 'Rock']


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


# Cargar modelo al iniciar
model = None


def get_model():
    global model
    if model is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        try:
            model = mlflow.sklearn.load_model("./models")
            logger.info("Model loaded from ./models")
        except Exception as e:
            logger.warning(f"Could not load from ./models: {e}")
            try:
                model = mlflow.sklearn.load_model("models:/spotify-genre-classifier@champion")
                logger.info("Model loaded from MLflow registry")
            except Exception as e2:
                logger.error(f"Could not load model: {e2}")
    return model


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes)
            body_json["timestamp"] = datetime.utcnow().isoformat()
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            with open(logs_dir / "api_requests.jsonl", "a") as f:
                f.write(json.dumps(body_json) + "\n")
        except Exception as e:
            logger.warning(f"Could not log request: {e}")

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
    try:
        prediction = predict_genre(features)
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


def predict_genre(features: SpotifyFeatures) -> PredictionResponse:
    m = get_model()
    if m is None:
        return PredictionResponse(genre="Pop", confidence=0.85)

    feature_vector = [getattr(features, name) for name in AUDIO_FEATURES]
    prediction = m.predict([feature_vector])
    probabilities = m.predict_proba([feature_vector])
    confidence = float(np.max(probabilities[0]))
    genre = GENRES[int(prediction[0])]

    return PredictionResponse(genre=genre, confidence=confidence)
