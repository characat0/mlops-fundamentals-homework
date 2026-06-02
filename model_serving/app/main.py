from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import logging
import os
from pathlib import Path

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpotifyFeatures(BaseModel):
    """
    Spotify audio features for genre classification.

    Based on the 550k Spotify Songs dataset from Kaggle.
    All values are normalized by Spotify to ranges [0, 1] except where noted.
    """
    danceability: float  # 0-1: How suitable for dancing
    energy: float  # 0-1: Intensity and activity
    key: int  # 0-11: Pitch class (C to B)
    loudness: float  # dB: Overall loudness
    mode: int  # 0-1: Major (1) or Minor (0)
    speechiness: float  # 0-1: Presence of spoken words
    acousticness: float  # 0-1: How acoustic
    instrumentalness: float  # 0-1: Likelihood of instrumental
    liveness: float  # 0-1: Presence of audience
    valence: float  # 0-1: Musical positiveness
    tempo: float  # BPM: Beats per minute
    duration_ms: int  # Milliseconds: Song length


class PredictionResponse(BaseModel):
    genre: str
    confidence: float = 0.0


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: SpotifyFeatures) -> PredictionResponse:
    """
    Predict Spotify track genre from audio features.

    Logs incoming requests to logs/api_requests.jsonl for drift monitoring.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    request_log_path = logs_dir / "api_requests.jsonl"

    request_data = {
        **features.model_dump(),
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }

    with open(request_log_path, "a") as f:
        f.write(json.dumps(request_data) + "\n")

    logger.info(f"Logged prediction request: {request_data}")

    try:
        prediction = predict_genre(features)
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


def predict_genre(features: SpotifyFeatures) -> PredictionResponse:
    """
    TODO: Load the MLflow model (baked into the container) and perform inference.

    For now, returns a placeholder response.
    In production, load the model like:
        import mlflow
        model = mlflow.sklearn.load_model("models:/champion@champion/production")
        prediction = model.predict([features.model_dump().values()])
    """
    return PredictionResponse(genre="Pop", confidence=0.85)
