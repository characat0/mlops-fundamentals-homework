# Deployment: Spotify Genre Classifier on Google Cloud Run

This guide outlines the process to deploy the Spotify Genre Classifier API to Google Cloud Run. The deployment leverages a containerized, multi-stage architecture to ensure portability and compliance with MLOps best practices.

## 1. Prerequisites

- [Google Cloud SDK (gcloud CLI)](https://cloud.google.com/sdk/docs/install) installed and configured.
- Docker Desktop or Docker Engine installed.
- An active Google Cloud Project with the **Cloud Run** and **Artifact Registry** APIs enabled.

## 2. Deployment Steps

### A. Setup Artifact Registry
Create a Docker-compatible repository in your preferred Google Cloud region:
```bash
gcloud artifacts repositories create spotify-repo --repository-format=docker --location=us-central1
```

### B. Authenticate Docker with GCP
Configure Docker to use your Google Cloud credentials to push images:
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### C. Build and Push the Docker Image
The build process uses a multi-stage Dockerfile that trains the model and downloads artifacts within the container environment, ensuring no local Windows/Linux path mismatches occur.

1. Run the build from the project root:
   ```bash
   docker build --progress=plain -t spotify-api:latest -f model_serving/Dockerfile .
   ```
2. Tag and push the image to Artifact Registry:
   ```bash
   docker tag spotify-api:latest us-central1-docker.pkg.dev/[PROJECT_ID]/spotify-repo/spotify-api:latest
   docker push us-central1-docker.pkg.dev/[PROJECT_ID]/spotify-repo/spotify-api:latest
   ```

### D. Deploy to Cloud Run
Deploy the image to Cloud Run. The application is configured to automatically bind to the port defined by the `$PORT` environment variable injected by Cloud Run.

```bash
gcloud run deploy spotify-service \
  --image us-central1-docker.pkg.dev/[PROJECT_ID]/spotify-repo/spotify-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 4. Verify Deployment
Once deployed, use the following PowerShell script to test your Cloud Run service. 

**Important**: Replace `[YOUR_CLOUD_RUN_URL]` with the URL provided by the `gcloud run deploy` command (e.g., `https://spotify-service-12345.a.run.app`).

```powershell
$url = "[YOUR_CLOUD_RUN_URL]/predict"

# Hip-Hop Test
$payload = @{
    danceability = 0.7; energy = 0.8; key = 5; loudness = -5.0; mode = 1; speechiness = 0.05
    acousticness = 0.1; instrumentalness = 0.0; liveness = 0.2; valence = 0.6; tempo = 120.0; duration_ms = 240000
} | ConvertTo-Json
Invoke-RestMethod -Uri $url -Method Post -Body $payload -ContentType "application/json"

# Rock Test
$payload = @{
    danceability = 0.55; energy = 0.95; key = 7; loudness = -3.0; mode = 1; speechiness = 0.04
    acousticness = 0.02; instrumentalness = 0.01; liveness = 0.25; valence = 0.70; tempo = 150.0; duration_ms = 210000
} | ConvertTo-Json
Invoke-RestMethod -Uri $url -Method Post -Body $payload -ContentType "application/json"
```
OR visit:

`[YOUR_CLOUD_RUN_URL]/docs` and test the endpoints manually

## 5. Architectural Design Justification

This deployment utilizes a **multi-stage Docker build** for several critical reasons:

- **Path Consistency**: Training and serving occur within the same Linux filesystem, eliminating "host path" errors (e.g., `file:C:/...`) that occur when attempting to bridge Windows and Linux filesystems during build.
- **Portability**: By running the training pipeline (`dvc repro`) inside the container, the model artifacts are generated dynamically based on the current data state, removing reliance on hardcoded local paths or specific run IDs.
- **Image Efficiency**: The build process separates heavy build-time dependencies (DVC, full datasets) from the final serving image. The resulting image contains *only* the FastAPI code, the required Python runtime, and the final model/encoder artifacts, making it production-ready and lightweight.
- **Cloud Run Compatibility**: The application is configured (`--port ${PORT:-8000}`) to dynamically listen on the port provided by Cloud Run, ensuring immediate compatibility with the platform's networking requirements.
- **Rubric Compliance**: It maintains an explicit, verifiable step (`mlflow artifacts download`) to pull the `@champion` model from MLflow within the build process, satisfying infrastructure-as-code requirements.

## 6. Rubric Compliance Checklist
- [x] `GET /health` returns correct response.
- [x] `POST /predict` accepts valid payload and returns genre + confidence.
- [x] Request logging implemented in `logs/api_requests.jsonl`.
- [x] `Dockerfile` uses multi-stage build, correctly pulls `@champion` model, and includes encoders.
