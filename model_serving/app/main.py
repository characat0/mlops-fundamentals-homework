import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator:
    """Pre-load model artifacts at startup so the first request is fast."""
    logger.info(f"Loading model from {MODELS_DIR}...")
    _load_artifacts()
    logger.info("Model loaded — ready to serve predictions.")
    yield


app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0", lifespan=lifespan)

_BASE_DIR = Path(__file__).resolve().parent.parent
_REMOTE_MODELS = _BASE_DIR / "models" / "remote"
_LOCAL_MODELS = _BASE_DIR / "models" / "local"

# Prefer models/remote/ (downloaded from MLflow at build-time) over
# models/local/ (committed fallback). This allows the Dockerfile to
# pull a fresh @champion while keeping the local copy as safety net.
MODELS_DIR = _REMOTE_MODELS if (_REMOTE_MODELS / "model.ubj").is_file() else _LOCAL_MODELS

LOGS_DIR = _BASE_DIR / "logs"
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
    Load the champion model and preprocessing artifacts.

    Resolution order:
      1. models/remote/ — fresh download from MLflow (docker build with
         DOWNLOAD_MODEL=true)
      2. models/local/  — committed fallback (always present in the repo)

    The model is loaded directly via xgboost (no mlflow runtime needed).
    """
    if not (MODELS_DIR / "label_encoder.joblib").is_file():
        raise FileNotFoundError(
            f"label_encoder.joblib not found in {MODELS_DIR}. "
            "Did the build step download the model artifacts?"
        )
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_order = joblib.load(MODELS_DIR / "feature_order.joblib")

    # Load XGBoost model directly from the .ubj file (no mlflow needed).
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "model.ubj"))

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
    # XGBoost handles scaling internally — no scaler needed for tree models.
    predicted_class = int(model.predict(feature_vector)[0])
    predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feature_vector)[0]
        confidence = float(np.max(proba))
    return PredictionResponse(genre=predicted_genre, confidence=round(confidence, 4))
