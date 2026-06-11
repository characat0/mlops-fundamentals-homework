from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import logging
from pathlib import Path
from datetime import datetime
import mlflow
import numpy as np
import pandas as pd
import traceback

app = FastAPI(title="Spotify Genre Classifier API", version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Modelo Pydantic con las features del dataset
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

# ✅ Middleware para loguear requests a /predict
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes.decode("utf-8"))
            body_json["timestamp"] = datetime.utcnow().isoformat()

            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            with open(logs_dir / "api_requests.jsonl", "a") as f:
                f.write(json.dumps(body_json) + "\n")
        except Exception as e:
            logger.error(f"Logging failed: {e}")

        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request = Request(request.scope, receive)

    response = await call_next(request)
    return response

# ✅ Endpoint de salud
@app.get("/health")
def health():
    return {"status": "healthy"}

# ✅ Endpoint de predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(features: SpotifyFeatures) -> PredictionResponse:
    try:
        return predict_genre(features)
    except Exception as e:
        logger.error("Prediction failed:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed")

# ✅ Lógica de predicción con MLflow PyFunc
def predict_genre(features: SpotifyFeatures) -> PredictionResponse:
    # ⚠️ Cargar modelo champion desde ./models (no ./models/model)
    model = mlflow.pyfunc.load_model("./models")

    # Convertir features a DataFrame
    df = pd.DataFrame([features.dict()])

    # Predecir
    prediction = model.predict(df)

    # ⚠️ Si el modelo no soporta predict_proba, dejamos confidence = 0.0
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(df)
            confidence = float(np.max(probabilities[0]))
        except Exception:
            pass

    predicted_genre = str(prediction[0])
    return PredictionResponse(genre=predicted_genre, confidence=confidence)
