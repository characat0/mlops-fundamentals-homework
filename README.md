# End-to-End MLOps Homework: Spotify Genre Classification & Drift Monitoring

Welcome to the final assignment for the MLOps Fundamentals and Practice course!

In this homework, you will implement a complete, production-ready MLOps pipeline for music audio data. You will manage data versioning, orchestrate experiments, deploy an API, and monitor for data drift.

## Dataset

**550k Spotify Songs** from Kaggle:
- **550,000 songs** with complete metadata and audio features
- **Audio features**: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms
- **Target**: Genre (10 main categories: Rock, Pop, Electronic, Folk, Country, Hip-Hop, R&B, Jazz, Blues, Classical)
- **Temporal data**: Release year (1960-2024) for simulating production drift
- **Download**: https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres
- **Note**: Requires Kaggle API authentication (see Setup section)

The task is to **classify music genre** from audio features. The release years are used to simulate real-world data drift: train on pre-2010 data (CD/iTunes era), test on post-2010 data (Spotify/streaming era). All 12 audio features show statistically significant drift across this boundary.

## Project Structure (Monorepo)

```text
Homework_E2E/
├── .github/workflows/       # CI/CD pipelines
├── data_pipeline/           # DVC orchestrated ML training pipeline
│   ├── src/                 # Scripts: download, process, train, evaluate
│   ├── tests/               # Unit tests for the pipeline steps
│   ├── dvc.yaml             # Pipeline definition (download → process → train → evaluate)
│   ├── params.yaml          # Hyperparameters and data config
│   └── requirements.txt      # Python dependencies
├── model_serving/           # FastAPI application and Docker deployment
│   ├── app/                 # FastAPI code
│   ├── tests/               # API integration tests
│   ├── Dockerfile           # Container definition
│   └── requirements.txt      # Python dependencies
└── drift_monitoring/        # Scripts for offline batch drift detection
    ├── src/
    └── requirements.txt
```

## Prerequisites

Before you start, ensure you have:
- **Python 3.9+** installed (`python --version`)
- **Git** installed and configured
- **Kaggle account** (free at https://www.kaggle.com)
  - Download API credentials from https://www.kaggle.com/settings/account
  - Save to `~/.kaggle/kaggle.json` (or follow `kaggle auth` prompts)
- **~2-3 GB free disk space** (for dataset + models + MLflow artifacts)
- **~15-20 minutes** for initial setup (includes ~5-10 min Kaggle download, ~10 min DVC repro)

---

## Setup

### 1. Clone & Install
```bash
git clone <your-fork-url>
cd Homework_E2E

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r data_pipeline/requirements.txt
pip install -r model_serving/requirements.txt
pip install -r drift_monitoring/requirements.txt

# Install Kaggle API for dataset download
pip install kaggle
```

### 2. Download Dataset
```bash
# Authenticate with Kaggle (requires credentials from https://www.kaggle.com/settings/account)
kaggle auth

# Download the dataset (saves to Kaggle's default directory, usually ~/.cache/kaggle/datasets/)
# Then ensure the CSV is placed in the data_pipeline/ directory
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env to set MLFLOW_TRACKING_URI (default: http://localhost:5000)
source .env
```

### 4. Start MLflow Server (in a separate terminal)
```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Your Tasks

### 1. Data Pipeline & Orchestration (DVC + MLflow)
Located in `data_pipeline/`.

#### 1.1 Download (`src/download.py`)
- **Status**: Skeleton provided
- **TODO**: Load the Kaggle CSV file (already downloaded) and prepare raw data
- **Input**: `params.yaml` with `data.source_path` (path to the downloaded CSV)
- **Output**: `data/raw.csv` (all columns from Kaggle dataset)
- **Notes**: The Kaggle dataset is in CSV format. Keep all audio features and the genre column.
- **Test**: `pytest data_pipeline/tests/test_process.py`

#### 1.2 Process (`src/process.py`)
- **Status**: Skeleton provided with implementation guidance
- **TODO**: Complete the function
- **Input**: Raw dataset, `year_threshold` from `params.yaml` (default 2010)
- **Output**: 
  - `data/train.csv` (year ≤ 2010) — Pre-streaming era data for model training
  - `data/prod_sim.csv` (year > 2010) — Streaming era data for drift detection
- **Key**: This is a **temporal split**, not random. You will detect drift: pre-2010 music (CD/iTunes) vs post-2010 (Spotify) have significantly different audio feature distributions.
- **Columns**: Include all audio features (danceability, energy, key, etc.), genre (target), and year (for reference)

#### 1.3 Train (`src/train.py`)
- **Status**: Skeleton provided
- **TODO**: Train **two different model types**: Logistic Regression + XGBoost
- **Requirements**:
  - Load training data from `data/train.csv`
  - **Target**: `genre` column (10-class classification)
  - **Features**: All audio features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms)
  - Load hyperparameters from `params.yaml`
  - Log to MLflow:
    - Parameters (e.g., `C`, `max_depth`, `learning_rate`)
    - Metrics (e.g., accuracy, precision, recall, F1)
    - Model artifacts using `mlflow.sklearn.log_model()` and `mlflow.xgboost.log_model()`
  - Create separate runs for each model type
  - Note: Scale features for Logistic Regression; XGBoost handles scaling internally

#### 1.4 Evaluate (`src/evaluate.py`)
- **Status**: Skeleton provided
- **TODO**: Programmatically select the best model and register it
- **Requirements**:
  - Query MLflow API to find all runs
  - Compare by accuracy metric
  - Register the best model in MLflow Model Registry
  - Assign it the alias `@champion` (students will reference this in Docker)
  - Output metrics summary to `metrics.json`

#### 1.5 DVC Pipeline (`dvc.yaml`)
- **Status**: Partially complete with TODOs addressed
- **What to do**: Run the pipeline
  ```bash
  cd data_pipeline
  dvc repro
  ```

### 2. Model Serving (FastAPI + Docker)
Located in `model_serving/`.

#### 2.1 API Implementation (`app/main.py`)
- **Status**: Skeleton with Pydantic models provided
- **TODO**: Implement `predict_genre()` function
- **Endpoints**:
  - `GET /health` → Returns `{"status": "healthy"}` (already implemented)
  - `POST /predict` → Accepts audio features (SpotifyFeatures), returns predicted genre
- **Requirements**:
  - Load the MLflow model registered with `@champion` alias
  - Perform inference on the 12 audio features
  - Return the predicted genre (one of: Rock, Pop, Electronic, Folk, Country, Hip-Hop, R&B, Jazz, Blues, Classical)
  - Request logging is **already implemented** (logs to `logs/api_requests.jsonl`)
  - Handle errors gracefully with HTTP 500 if prediction fails

#### 2.2 Tests (`tests/test_api.py`)
- **Status**: Real tests provided (no placeholder assertions)
- **What to do**: Ensure they pass
  ```bash
  cd model_serving
  pytest tests/
  ```

#### 2.3 Dockerfile (`Dockerfile`)
- **Status**: Skeleton with comments on MLflow model pulling
- **TODO**: Add a build step to pull the `@champion` aliased model from MLflow
- **Example**:
  ```dockerfile
  ARG MLFLOW_TRACKING_URI=http://mlflow:5000
  RUN mlflow models download -m models:/champion@champion -d ./models --no-directory
  ```
- **Note**: The model will be saved to the container's `./models` directory

### 3. Drift Monitoring (Batch)
Located in `drift_monitoring/`.

#### 3.1 Drift Analysis (`src/analyze_drift.py`)
- **Status**: Skeleton provided
- **TODO**: Compare training data distribution vs production logs
- **Input**:
  - Training data: `data_pipeline/data/train.csv` (from DVC)
  - Production logs: `logs/api_requests.jsonl` (from API)
- **Output**: A drift report or alert
- **Suggested approach**:
  - Use statistical tests (Kolmogorov-Smirnov, Chi-square)
  - Compare feature distributions
  - Flag if drift exceeds a threshold
  - Save results to `drift_report.json`

### 4. CI/CD (GitHub Actions)
Located in `.github/workflows/ci.yml`.

#### 4.1 Pipeline Configuration
- **Status**: Fixed to install all requirements
- **What runs on every PR**:
  1. `flake8 .` — Linting
  2. `pytest data_pipeline/tests` — Data pipeline tests
  3. `pytest model_serving/tests` — API tests
- **Make sure it passes**: Green checkmark = pipeline works!

## Submission Checklist

- [ ] All functions in `src/*.py` are implemented (no `pass` statements)
- [ ] `params.yaml` has realistic hyperparameters
- [ ] `dvc repro` runs without errors in `data_pipeline/`
- [ ] `pytest` passes for both `data_pipeline/tests/` and `model_serving/tests/`
- [ ] `flake8 .` shows no major style violations
- [ ] MLflow server has runs logged with metrics and models
- [ ] Best model is registered with `@champion` alias
- [ ] API returns predictions with valid payloads
- [ ] API logs requests to `logs/api_requests.jsonl`
- [ ] Dockerfile builds successfully
- [ ] Drift monitoring script runs without errors
- [ ] GitHub Actions workflow passes (green checkmark on PR)
- [ ] All TODO comments in your code are addressed or justified

## Troubleshooting

### CI/CD Pipeline Failed?

**Linter errors** (`flake8` failed):
```bash
# Run locally to see issues before pushing
flake8 .

# Common fixes:
# - Remove trailing whitespace
# - Fix indentation (use 4 spaces, not tabs)
# - Remove unused imports
# - Keep lines under 100 characters
```

**Process tests fail**:
```bash
# Check data pipeline locally
cd data_pipeline
dvc repro         # Run the full pipeline
dvc dag           # Check the pipeline structure
ls -la data/      # Verify outputs exist

# Common issues:
# - Missing data/raw.csv? Download.py not working
# - Missing data/train.csv? Process.py not splitting correctly
# - Check year threshold (2010) is used correctly
```

**API tests fail**:
```bash
# Test locally
cd model_serving
pytest tests/ -v

# Common issues:
# - SpotifyFeatures fields don't match test payload
# - /predict endpoint doesn't handle missing fields (should return 422)
# - Response format doesn't include "genre" and "confidence" keys
```

**MLflow model registration fails**:
```bash
# Verify MLflow is running
curl http://localhost:5000

# Check registered models
mlflow models list

# Check model URI format
# Should be: runs:/{run_id}/model (not models:/)
```

---

## Useful Commands

```bash
# DVC
cd data_pipeline
dvc repro                        # Run the full pipeline
dvc dag                          # Visualize the DAG
dvc status                       # Check what needs to be rerun

# MLflow
mlflow server --host 0.0.0.0 --port 5000    # Start MLflow UI (http://localhost:5000)
mlflow models list                            # List registered models

# Testing
pytest data_pipeline/tests -v
pytest model_serving/tests -v
flake8 .

# API
cd model_serving
uvicorn app.main:app --reload --port 8000

# Docker
docker build -t spotify-api:latest .
docker run -p 8000:8000 spotify-api:latest
```

## Quick Troubleshooting

- **`flake8` errors**: Run `flake8 .` locally, fix style issues (whitespace, imports, line length)
- **Process tests fail**: Ensure CSV is in correct path and `year` column exists
- **MLflow runs not showing up**: Verify MLflow server is running with `mlflow server --host 0.0.0.0 --port 5000`
- **API tests fail**: Check that `SpotifyFeatures` field names match test payload (12 audio features)
- **Dockerfile build fails**: Ensure `@champion` model download step uses correct MLflow URI

---

## Notes for Students

1. **Genre Classification + Drift**: The task is to classify song **genre** (target) from audio features. The release **year** is NOT the target — it's used to create a temporal train/test split that simulates real data drift.

   **The 2010 boundary marks the Streaming Era Shift:**
   - **Pre-2010 (Training)**: CD/iTunes era — longer songs, more acoustic, higher emotional valence
   - **Post-2010 (Production)**: Spotify/Apple Music era — shorter, punchier, more electronic, heavily compressed
   
   **All 12 features show statistically significant drift:**
   - 🔊 Loudness: +1.56 dB (loudness wars & compression)
   - 🎸 Acousticness: -5.75% (more synth, less acoustic)
   - 😔 Valence: -6.5% (moodier music)
   - ⚡ Energy: +4.3% (more intense production)
   - ⏱️ Duration: -8.4 sec (streaming optimization)

2. **Audio Features**: The 12 Spotify audio features represent objective measurements of the audio:
   - Danceability: How suitable for dancing (0-1)
   - Energy: Intensity and activity (0-1)
   - Key: Pitch class (0-11)
   - Loudness: Overall loudness in dB
   - Mode: Major (1) or minor (0)
   - Speechiness, Acousticness, Instrumentalness, Liveness: Presence of these elements (0-1)
   - Valence: Musical positiveness (0-1)
   - Tempo: Beats per minute
   - Duration: Song length in milliseconds

3. **MLflow Aliases**: The `@champion` alias is a way to version models. The Dockerfile will pull exactly this alias, ensuring the container always runs the approved best model.

4. **Drift Monitoring**: In production, data drift detection prevents silent model degradation. You're comparing the distribution of incoming requests (API logs) to the training data distribution. Significant drift signals that the model may be underperforming.

## Grading Rubric

See `GRADING_RUBRIC.md` for detailed point breakdowns (20 points total).

---

## Helpful Resources

- [DVC Docs](https://dvc.org/doc)
- [MLflow Docs](https://mlflow.org/docs/latest/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [Million Song Dataset Paper](https://www.ee.columbia.edu/~dpwe/pubs/BertEWL11-msd.pdf)

Good luck! 🚀
