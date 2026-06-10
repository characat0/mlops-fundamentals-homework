import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
API_LOG_PATH = LOGS_DIR / "api_requests.jsonl"

AUDIO_FEATURES = [
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


class SpotifyFeatures(BaseModel):
    danceability: float = Field(..., ge=0.0, le=1.0)
    energy: float = Field(..., ge=0.0, le=1.0)
    key: int = Field(..., ge=0, le=11)
    loudness: float
    mode: int = Field(..., ge=0, le=1)
    speechiness: float = Field(..., ge=0.0, le=1.0)
    acousticness: float = Field(..., ge=0.0, le=1.0)
    instrumentalness: float = Field(..., ge=0.0, le=1.0)
    liveness: float = Field(..., ge=0.0, le=1.0)
    valence: float = Field(..., ge=0.0, le=1.0)
    tempo: float = Field(..., gt=0.0)
    duration_ms: int = Field(..., gt=0)


class PredictionResponse(BaseModel):
    genre: str
    confidence: float = 0.0


@lru_cache(maxsize=1)
def _load_artifacts():
    """
    Load the champion model and the preprocessing artifacts (label encoder,
    scaler, feature order) from the local ./models directory.
    The Dockerfile bakes these in at build time via the MLflow download step
    and copies the preprocessors next to them.
    """
    if not (MODELS_DIR / "label_encoder.joblib").is_file():
        raise FileNotFoundError(
            f"label_encoder.joblib not found in {MODELS_DIR}. "
            "Did the build step download the model artifacts?"
        )
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_order = joblib.load(MODELS_DIR / "feature_order.joblib")
    model = mlflow.pyfunc.load_model(str(MODELS_DIR))
    return model, label_encoder, scaler, feature_order


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path != "/predict" or request.method != "POST":
        return await call_next(request)
    body_bytes = await request.body()
    try:
        payload = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        payload = {"_raw": body_bytes.decode("utf-8", errors="replace")}
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with API_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    async def receive():
        return {"type": "http.request", "body": body_bytes}

    request = Request(request.scope, receive)
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: SpotifyFeatures) -> PredictionResponse:
    try:
        return predict_genre(features)
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail="Prediction failed")


def predict_genre(features: SpotifyFeatures) -> PredictionResponse:
    model, label_encoder, scaler, feature_order = _load_artifacts()
    feature_vector = np.array(
        [[getattr(features, name) for name in feature_order]],
        dtype=float,
    )
    impl = getattr(model, "_model_impl", model)
    underlying = getattr(impl, "xgb_model", impl)
    is_tree_based = underlying.__class__.__name__ == "XGBClassifier"
    X_to_use = feature_vector if is_tree_based else scaler.transform(feature_vector)
    predicted_class = int(underlying.predict(X_to_use)[0])
    predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
    confidence = 0.0
    if hasattr(underlying, "predict_proba"):
        proba = underlying.predict_proba(X_to_use)[0]
        confidence = float(np.max(proba))
    return PredictionResponse(genre=predicted_genre, confidence=round(confidence, 4))
