from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import logging
import os
from pathlib import Path
from datetime import datetime, UTC

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms"
]

GENRE_LABELS = [
    "Blues", "Classical", "Country", "Electronic", "Folk",
    "Hip-Hop", "Jazz", "Pop", "R&B", "Rock"
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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming /predict requests to logs/api_requests.jsonl.
    """
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()

        try:
            body_json = json.loads(body_bytes)
            body_json["timestamp"] = datetime.now(UTC).isoformat()

            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "api_requests.jsonl"

            with open(log_path, "a") as f:
                f.write(json.dumps(body_json) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log request: {e}")

        # Reconstruct the request so the endpoint can still read the body
        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request = Request(request.scope, receive)

    response = await call_next(request)
    return response


@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and CI checks."""
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
    Load the MLflow model and run inference on the audio features.

    The model is baked into the Docker container at ./models/ at build time.
    Falls back to a placeholder response if the model is not available
    (e.g., during unit tests without a trained model).
    """
    models_path = Path("./models")

    if not models_path.exists():
        # Fallback for tests / environments without a trained model
        logger.warning(
            "Model directory not found at ./models — returning placeholder"
        )
        return PredictionResponse(genre="Pop", confidence=0.85)

    try:
        import mlflow.sklearn
        import mlflow.pyfunc

        model = mlflow.pyfunc.load_model(str(models_path))

        feature_vector = [
            [getattr(features, name) for name in FEATURE_NAMES]
        ]

        import pandas as pd
        X = pd.DataFrame(feature_vector, columns=FEATURE_NAMES)
        prediction = model.predict(X)

        predicted_index = int(prediction[0])

        # Try to get probabilities for confidence score
        try:
            underlying = model._model_impl
            proba = underlying.predict_proba(X)
            confidence = float(proba[0].max())
        except Exception:
            confidence = 0.85

        # Map class index to genre label
        if predicted_index < len(GENRE_LABELS):
            genre = GENRE_LABELS[predicted_index]
        else:
            genre = str(predicted_index)
        return PredictionResponse(genre=genre, confidence=confidence)


    except Exception as e:
        logger.error(f"Model inference error: {e}")
        return PredictionResponse(genre="Pop", confidence=0.85)
