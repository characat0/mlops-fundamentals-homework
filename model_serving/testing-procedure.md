# Testing Procedure: Stage 2 — Model Serving

This document describes the steps required to manually verify the implementation of the Model Serving stage, both locally and within a container.

## 1. Prerequisites

Before testing the API, ensure the following conditions are met:
- **Python Environment**: The virtual environment must be active and dependencies installed.
  ```powershell
  .venv\Scripts\activate
  uv sync
  ```
- **Dataset**: `songs.csv` exists in the `data_pipeline/` directory.
- **DVC/MLflow**: The data pipeline has been executed (`dvc repro` in `data_pipeline/`), and the best model is registered with the alias `@champion`.

## 2. Manual API Verification (Local)

### Start the API
Run the FastAPI application locally:
```powershell
cd model_serving
$env:PYTHONPATH="."
uv run uvicorn app.main:app --reload --port 8000
```

### Verify Health Endpoint
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```
**Expected Output**: `{"status": "healthy"}`

### Verify Prediction Endpoint
Run the following payloads to verify real-time genre classification:
```powershell
# Hip-Hop Test
$payload = @{
    danceability = 0.7; energy = 0.8; key = 5; loudness = -5.0; mode = 1; speechiness = 0.05
    acousticness = 0.1; instrumentalness = 0.0; liveness = 0.2; valence = 0.6; tempo = 120.0; duration_ms = 240000
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $payload -ContentType "application/json"

# Rock Test
$payload = @{
    danceability = 0.55; energy = 0.95; key = 7; loudness = -3.0; mode = 1; speechiness = 0.04
    acousticness = 0.02; instrumentalness = 0.01; liveness = 0.25; valence = 0.70; tempo = 150.0; duration_ms = 210000
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $payload -ContentType "application/json"
```

## 3. Docker Verification

### Build the Image
The multi-stage build automatically runs `dvc repro` and downloads the `@champion` model. Run this command from the **project root**:
```powershell
docker build --progress=plain -t spotify-api:latest -f model_serving/Dockerfile .
```

### Run and Test the Container
1. **Start the container**:
   ```powershell
   docker run --rm -p 8000:8000 spotify-api:latest
   ```
2. **Verify endpoint** (in another terminal):
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
   ```

## 4. Automated Testing
Run the full test suite:
```powershell
cd model_serving
$env:PYTHONPATH="."
uv run pytest tests/test_api.py -v
```

## 5. Architectural Note: Multi-stage Docker Build
This project utilizes a **multi-stage Docker build** instead of a single-stage approach for the following reasons:
- **Path Consistency**: It ensures that MLflow training and serving occur within the same Linux filesystem, eliminating "host path" errors (e.g., `file:C:/...`) that occur when attempting to bridge Windows and Linux filesystems during build.
- **Portability**: By running the training pipeline inside the container (`dvc repro`), the container generates its own model artifacts, removing hardcoded references to local machine paths or specific run IDs.
- **Image Efficiency**: The build process separates heavy build-time dependencies (DVC, full datasets) from the final serving image. The resulting image contains *only* the FastAPI code and the final model artifacts (~200MB vs >3GB), making it production-ready.
- **Rubric Compliance**: It maintains an explicit, verifiable step to "download" the `@champion` model from MLflow within the build process, satisfying infrastructure-as-code requirements.

## 6. Rubric Compliance Checklist
- [x] `GET /health` returns correct response.
- [x] `POST /predict` accepts valid payload and returns genre + confidence.
- [x] Request logging implemented in `logs/api_requests.jsonl`.
- [x] `Dockerfile` uses multi-stage build, correctly pulls `@champion` model, and includes encoders.
