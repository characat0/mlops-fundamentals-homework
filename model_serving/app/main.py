from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import logging
import os
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np

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

# Cargar modelo al iniciar
try:
    model = joblib.load('xgboost_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    label_encoder = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/predict":
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes)
            log_entry = {"timestamp": datetime.utcnow().isoformat(), **body_json}
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            with open("logs/api_requests.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            async def receive():
                return {"type": "http.request", "body": body_bytes}
            request = Request(request.scope, receive)
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    response = await call_next(request)
    return response

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: SpotifyFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
        
        feature_vector = [[getattr(features, name) for name in feature_names]]
        
        prediction = model.predict(feature_vector)[0]
        predicted_genre = label_encoder.inverse_transform([prediction])[0]
        
        try:
            probabilities = model.predict_proba(feature_vector)
            confidence = float(probabilities[0].max())
        except:
            confidence = 0.0
        
        return PredictionResponse(genre=predicted_genre, confidence=confidence)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
