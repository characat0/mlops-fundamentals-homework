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
            payload = json.loads(body_bytes)
            payload["timestamp"] = datetime.utcnow().isoformat()

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "api_requests.jsonl"

            with open(log_file, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.error(f"Failed to log request: {str(e)}")

        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request = Request(request.scope, receive)

    response = await call_next(request)
    return response


# TODO: Implement the GET /health endpoint.
#   It should return {"status": "healthy"} with a 200 status code.
#   This is used by load balancers and CI checks to verify the API is up.
@app.get("/health")
def health_check():
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
    return PredictionResponse(genre="Pop", confidence=0.85)
