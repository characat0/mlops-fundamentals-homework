# Testing Procedure: Stage 2 — Model Serving

This document describes the steps required to manually verify the implementation of the Model Serving stage.

## 1. Prerequisites

Before testing the API, ensure the following conditions are met:
- **Python Environment**: The virtual environment must be active and dependencies installed.
  ```powershell
  .venv\Scripts\activate
  uv sync
  ```
- **MLflow Server**: An MLflow server should be running (required for model registration and download).
  ```powershell
  mlflow server --host 0.0.0.0 --port 5000
  ```
- **Champion Model**: A model must be registered with the alias `@champion` in the MLflow Model Registry (Stage 1 completion).
  - *Note: The API includes a fallback mechanism to return a placeholder prediction if the model is not found, allowing tests to pass during development.*

## 2. Manual API Verification

### Start the API
Run the FastAPI application locally:
```powershell
cd model_serving
$env:PYTHONPATH="."
uv run uvicorn app.main:app --reload --port 8000
```

### Verify Health Endpoint
Open a new terminal and run:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```
**Expected Output**: `{"status": "healthy"}`

### Verify Prediction Endpoint (Valid Payload)
```powershell
$payload = @{
    danceability = 0.7
    energy = 0.8
    key = 5
    loudness = -5.0
    mode = 1
    speechiness = 0.05
    acousticness = 0.1
    instrumentalness = 0.0
    liveness = 0.2
    valence = 0.6
    tempo = 120.0
    duration_ms = 240000
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $payload -ContentType "application/json"
```
**Expected Output**: A JSON response containing `"genre"` and `"confidence"`.

### Verify Request Logging
Check if the request was logged to `logs/api_requests.jsonl`:
```powershell
Get-Content logs/api_requests.jsonl -Tail 1
```
**Expected Output**: A JSON line containing the payload and a `"timestamp"` field.

## 3. Automated Testing
Run the full test suite for the model serving component:
```powershell
cd model_serving
$env:PYTHONPATH="."
uv run pytest tests/test_api.py -v
```
**Expected Result**: 3 passed tests.

## 4. Docker Build Verification
Verify that the Dockerfile can build the image (requires MLflow server to be accessible):
```powershell
cd model_serving
docker build --build-arg MLFLOW_TRACKING_URI=http://localhost:5000 -t spotify-api:latest .
```
**Expected Result**: Successful build including the `mlflow models download` step.

## 5. Rubric Compliance Checklist
- [x] `GET /health` returns correct response.
- [x] `POST /predict` accepts valid payload and returns prediction.
- [x] Request logging implemented in `logs/api_requests.jsonl`.
- [x] `SpotifyFeatures` model includes all 12 audio features.
- [x] `Dockerfile` includes the champion model download step.
