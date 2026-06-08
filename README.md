# End-to-End MLOps Homework: Spotify Genre Classification & Drift Monitoring

Welcome to the final assignment for the MLOps Fundamentals and Practice course!

In this homework, you will implement a complete, production-ready MLOps pipeline for music audio data. You will manage data versioning, orchestrate experiments, deploy an API, and monitor for data drift.

---

## How to Submit

Follow these steps exactly — your grade depends on a passing CI run on your Pull Request.

### Step 1 — Fork the repository

1. Go to the course repository on GitHub.
2. Click **Fork** (top-right corner) → **Create fork**.
3. This creates a copy under your GitHub account (e.g., `your-username/mlops-fundamentals-homework`).

### Step 2 — Clone your fork

```bash
git clone https://github.com/<your-username>/mlops-fundamentals-homework.git
cd mlops-fundamentals-homework
```

> Replace `<your-username>` with your actual GitHub username.

### Step 3 — Create a working branch

```bash
git checkout -b solution/<your-name>
# Example: git checkout -b solution/maria-garcia
```

### Step 4 — Implement the tasks

Complete all TODOs described in the [Implementation Checklist](#implementation-checklist) below. Commit your progress regularly:

```bash
git add .
git commit -m "feat: implement data pipeline process step"
git push origin solution/<your-name>
```

### Step 5 — Open a Pull Request

1. Go to your fork on GitHub.
2. Click **Compare & pull request** (GitHub will show this banner automatically after a push).
3. Set the **base repository** to the course repo and **base branch** to `main`.
4. Title your PR: `[Homework] <Your Full Name>` (e.g., `[Homework] Maria Garcia`).
5. In the description, paste your completed [Submission Checklist](GRADING_RUBRIC.md).
6. Click **Create pull request**.

> **The CI pipeline runs automatically on every push to your PR.** A green checkmark means your tests and linter pass — this is worth 1 point. See [§4.3 GitHub Actions](GRADING_RUBRIC.md#43-github-actions-1-point).

---

## Dataset

**550k Spotify Songs** from Kaggle:
- **550,000 songs** with complete metadata and audio features
- **File location**: Save as `songs.csv` in the `data_pipeline/` directory
- **Download**: https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres
- **Note**: Requires Kaggle API authentication (see Setup section)

### Audio Features
Your model should use these audio features as inputs:
- `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`
- `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`

**Target**: `genre` column (10 main categories: Rock, Pop, Electronic, Folk, Country, Hip-Hop, R&B, Jazz, Blues, Classical)

**Other columns** (metadata — you can choose to include or ignore):
- `id`, `name`, `album_name`, `artists`, `lyrics` (track metadata)
- `popularity`, `total_artist_followers`, `avg_artist_popularity`, `artist_ids`, `niche_genres` (popularity metrics)

### Temporal Split: The 2010 Streaming Era Boundary
The task is to **classify music genre** while detecting **data drift**:
- **Training Data** (year ≤ 2010): CD/iTunes era — longer songs, more acoustic, higher emotional valence
- **Production Data** (year > 2010): Spotify/streaming era — shorter, punchier, more electronic, heavily compressed
- **Why 2010?** This marks the launch of Spotify and shift to streaming-dominant music consumption
- **Data Drift**: Audio features show statistically significant drift across this boundary, simulating real-world model degradation

---

## Project Structure (Monorepo)

```text
mlops-fundamentals-homework/
├── .github/workflows/       # CI/CD pipelines
├── data_pipeline/           # DVC orchestrated ML training pipeline
│   ├── src/                 # Scripts: load, process, train, evaluate
│   ├── tests/               # Unit tests for the pipeline steps
│   ├── dvc.yaml             # Pipeline definition (load → process → train → evaluate)
│   ├── params.yaml          # Hyperparameters and data config
│   └── requirements.txt     # Python dependencies
├── model_serving/           # FastAPI application and Docker deployment
│   ├── app/                 # FastAPI code
│   ├── tests/               # API integration tests
│   ├── Dockerfile           # Container definition
│   └── requirements.txt     # Python dependencies
└── drift_monitoring/        # Scripts for offline batch drift detection
    ├── src/
    └── requirements.txt
```

---

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

### 1. Install dependencies
```bash
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
# Then place the CSV at: data_pipeline/songs.csv
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

---

## Implementation Checklist

Every item below maps to a specific TODO in the code. Complete them in order — each stage feeds the next. Point values reference the [GRADING_RUBRIC](GRADING_RUBRIC.md).

### Stage 1 — Data Pipeline (`data_pipeline/`) · *6 pts* · [Rubric §1](GRADING_RUBRIC.md#1-data-pipeline-6-points)

**`src/process.py` → `process_data()`**
- [x] Split `df` into `train_df` (year ≤ 2010) and `prod_df` (year > 2010)
- [x] Save both to CSV using `to_csv(..., index=False)`

**`src/train.py` → `train()`**
- [x] Encode `genre` labels with `LabelEncoder`
- [x] Scale features with `StandardScaler` (LogisticRegression only — XGBoost skips this)
- [x] Loop through `params['train']`, for each model:
  - Start an MLflow run (`mlflow.start_run(run_name=...)`)
  - Log hyperparameters (`mlflow.log_params(...)`)
  - Fit the model
  - Log accuracy metric (`mlflow.log_metric("accuracy", ...)`)
  - Save the model artifact (`mlflow.sklearn.log_model` or `mlflow.xgboost.log_model`)

**`src/evaluate.py` → `evaluate_and_register()`**
- [x] Call `client.create_model_version(name, source, run_id)` to register the best run
- [x] Call `client.set_registered_model_alias(name, "champion", version)` to tag it

---

### Stage 2 — Model Serving (`model_serving/`) · *5 pts* · [Rubric §2](GRADING_RUBRIC.md#2-model-serving-5-points)

**`app/main.py` → `SpotifyFeatures`**
- [x] Add the audio feature fields with correct types to the Pydantic model

**`app/main.py` → `GET /health`**
- [x] Implement the health endpoint returning `{"status": "healthy"}`

**`app/main.py` → `log_requests` middleware**
- [x] Read the request body and parse as JSON
- [x] Append a JSON line (with timestamp) to `logs/api_requests.jsonl`
- [x] Reconstruct the request before passing to `call_next`

**`app/main.py` → `predict_genre()`**
- [x] Load the MLflow model from `./models/` (baked in at Docker build time)
- [x] Extract the feature values from the `SpotifyFeatures` object
- [x] Run inference and return a `PredictionResponse` with genre and confidence

**`Dockerfile`**
- [x] Add `ARG MLFLOW_TRACKING_URI` and the `RUN mlflow models download` step to pull `@champion` into `./models/`

---

### Stage 3 — Drift Monitoring (`drift_monitoring/`) · *3 pts* · [Rubric §3](GRADING_RUBRIC.md#3-drift-monitoring-3-points)

**`src/analyze_drift.py` → `run_ks_analysis()`** *(shared by both modes)*
- [x] For each feature in `features_to_test`, run `scipy.stats.ks_2samp(train_values, prod_values)`
- [x] Flag drift if `p_value < 0.05`
- [x] Populate `drift_results["details"][feature]` with `ks_statistic`, `p_value`, `drift_detected`, `train_mean`, `prod_mean`
- [x] Append drifted feature names to `drift_results["drifted_features"]` and update `features_with_drift`

**Batch mode** (`--mode batch`): called by `analyze_batch_drift()` — loads `data/train.csv` and `data/prod_sim.csv`

**Online mode** (`--mode online`): called by `analyze_online_drift()` — loads `data/train.csv` and `logs/api_requests.jsonl`

---

## Your Tasks

### 1. Data Pipeline & Orchestration (DVC + MLflow) · [Rubric §1](GRADING_RUBRIC.md#1-data-pipeline-6-points)
Located in `data_pipeline/`.

#### 1.1 Load (`src/load.py`)
- **Status**: Provided — no changes needed
- **Input**: `songs.csv` in the `data_pipeline/` directory (downloaded manually from Kaggle)
- **Output**: `data/raw.csv` (all columns, no filtering)
- **Hash verification**: After placing your `songs.csv`, verify you have the correct file:
  ```bash
  cd data_pipeline
  dvc status songs.csv.dvc   # Should show: songs.csv: not in cache
                              # Then run: dvc repro
                              # After repro, dvc.lock will record the hash
  ```
  If the hash in `dvc.lock` matches `songs.csv.dvc`, you have the right file. See [§1.1 Dataset Integrity](GRADING_RUBRIC.md#11-dataset-integrity-1-point).

#### 1.2 Process (`src/process.py`)
- **Status**: Skeleton provided — implement the split and save logic (see TODOs in `process_data()`)
- **Input**: Raw dataset, `year_threshold` from `params.yaml` (**set to 2010** — not 2005)
- **Output**:
  - `data/train.csv` (year ≤ 2010) — Pre-streaming era data for model training
  - `data/prod_sim.csv` (year > 2010) — Streaming era data for drift detection
- **Key**: This is a **temporal split**, not random. You will detect drift: pre-2010 music (CD/iTunes) vs post-2010 (Spotify) have significantly different audio feature distributions.
  - **Important**: `year_threshold = 2010` marks the streaming era shift. This boundary is where statistically significant drift occurs.
- **Columns**: Include all audio features (danceability, energy, key, etc.), genre (target), and year (for reference)

#### 1.3 Train (`src/train.py`)
- **Status**: Skeleton provided
- **TODO**: Train **two different model types**: Logistic Regression + XGBoost
- **Requirements**:
  - Load training data from `data/train.csv`
  - **Target**: `genre` column (10-class classification)
  - **Features**: Audio features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms)
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

### 2. Model Serving (FastAPI + Docker) · [Rubric §2](GRADING_RUBRIC.md#2-model-serving-5-points)
Located in `model_serving/`.

#### 2.1 API Implementation (`app/main.py`)
- **Status**: Skeleton with Pydantic models provided
- **TODO**: Implement `predict_genre()` function
- **Endpoints**:
  - `GET /health` → Returns `{"status": "healthy"}` (already implemented)
  - `POST /predict` → Accepts audio features (SpotifyFeatures), returns predicted genre
- **Requirements**:
  - Load the MLflow model registered with `@champion` alias
  - Perform inference on the audio features
  - Return the predicted genre (one of: Rock, Pop, Electronic, Folk, Country, Hip-Hop, R&B, Jazz, Blues, Classical)
  - Request logging is **already implemented** (logs to `logs/api_requests.jsonl`)
  - Handle errors gracefully with HTTP 500 if prediction fails
- **Implementation guidance**: See `predict_genre()` docstring in `main.py` for detailed instructions including model loading example

#### 2.2 Tests (`tests/test_api.py`)
- **Status**: Real tests provided (no placeholder assertions)
- **What to do**: Ensure they pass
  ```bash
  cd model_serving
  pytest tests/
  ```

#### 2.3 Dockerfile (`Dockerfile`)
- **Status**: Skeleton provided
- **TODO**: Add an `ARG` and `RUN` step to download the `@champion` model from MLflow during build
- **Requirements**:
  - Accept `MLFLOW_TRACKING_URI` as a build argument (default: `http://localhost:5000`)
  - Use `mlflow models download` to pull the `@champion` model into `./models/` at build time
  - The container should be self-contained (no live MLflow server needed at runtime)
- **Hint**: See the TODO comment inside `Dockerfile` for the exact command syntax
- **Overriding MLflow URI**: When building in different environments:
  ```bash
  docker build --build-arg MLFLOW_TRACKING_URI=http://mlflow-host:5000 .
  ```

##### MLflow Model Loading

Implement the following in the `predict_genre()` function:
```python
mlflow.sklearn.load_model("models:/champion@champion/production")
```

**Important Networking Note**:
- **Local development** (MLflow on your machine): Use `http://localhost:5000` as the tracking URI
- **Docker internal network** (if using docker-compose): Use `http://mlflow:5000` for container-to-container communication
- **Docker build time**: The build process runs on your machine, so use `http://localhost:5000` to download the model
- **Docker runtime**: The container uses the URI passed at build time or set as an environment variable

If the model loading fails with "Connection refused" in the Docker container, verify that MLflow is accessible from the container's network context.

### 3. Drift Monitoring · [Rubric §3](GRADING_RUBRIC.md#3-drift-monitoring-3-points)
Located in `drift_monitoring/`.

The script supports two modes — run both to get the full picture.

#### 3.1 Batch Drift (`src/analyze_drift.py --mode batch`)
- **Status**: Skeleton provided — implement `run_ks_analysis()`
- **What it does**: Compares `data/train.csv` vs `data/prod_sim.csv` (the two temporal splits from `process.py`)
- **Input**:
  - `--train_data data_pipeline/data/train.csv`
  - `--prod_data data_pipeline/data/prod_sim.csv`
  - `--output batch_drift_report.json`
- **Expected result**: Significant drift detected — pre-2010 (CD/iTunes era) vs post-2010 (Spotify era) have very different audio distributions
- **TODO**: Implement the KS test loop in `run_ks_analysis()` — this function is shared with online mode, so you only write it once

```bash
cd drift_monitoring
python src/analyze_drift.py \
  --mode batch \
  --train_data ../data_pipeline/data/train.csv \
  --prod_data ../data_pipeline/data/prod_sim.csv \
  --output batch_drift_report.json
```

#### 3.2 Online Drift (`src/analyze_drift.py --mode online`)
- **Status**: Skeleton provided — reuses `run_ks_analysis()` from batch mode
- **What it does**: Compares `data/train.csv` vs live API request logs (`logs/api_requests.jsonl`)
- **Input**:
  - `--train_data data_pipeline/data/train.csv`
  - `--api_logs model_serving/logs/api_requests.jsonl`
  - `--output online_drift_report.json`
- **Prerequisite**: The API must be running and have received prediction requests (middleware logs them)

```bash
cd drift_monitoring
python src/analyze_drift.py \
  --mode online \
  --train_data ../data_pipeline/data/train.csv \
  --api_logs ../model_serving/logs/api_requests.jsonl \
  --output online_drift_report.json
```

### 4. CI/CD (GitHub Actions) · [Rubric §4](GRADING_RUBRIC.md#4-testing--cicd-4-points)
Located in `.github/workflows/ci.yml`.

#### 4.1 Pipeline Configuration
- **Status**: Fixed to install all requirements
- **What runs on every PR**:
  1. `flake8 .` — Linting
  2. `pytest data_pipeline/tests` — Data pipeline tests
  3. `pytest model_serving/tests` — API tests
- **Make sure it passes**: Green checkmark = pipeline works!

---

## Submission Checklist

Review the full point breakdown in [GRADING_RUBRIC](GRADING_RUBRIC.md) before submitting.

- [x] All functions in `src/*.py` are implemented (no `pass` statements)
- [x] `params.yaml` has realistic hyperparameters
- [x] `dvc repro` runs without errors in `data_pipeline/` · [§1.5](GRADING_RUBRIC.md#15-dvc-pipeline-05-points)
- [x] `pytest` passes for both `data_pipeline/tests/` and `model_serving/tests/` · [§4.1](GRADING_RUBRIC.md#41-unit-tests-2-points)
- [x] `flake8 .` shows no major style violations · [§4.2](GRADING_RUBRIC.md#42-code-quality-1-point)
- [x] MLflow server has runs logged with metrics and models · [§1.3](GRADING_RUBRIC.md#13-train-script-2-points)
- [x] Best model is registered with `@champion` alias · [§1.4](GRADING_RUBRIC.md#14-evaluate-script-1-point)
- [x] API returns predictions with valid payloads · [§2.1](GRADING_RUBRIC.md#21-api-implementation-3-points)
- [x] API logs requests to `logs/api_requests.jsonl` · [§2.1](GRADING_RUBRIC.md#21-api-implementation-3-points)
- [x] Dockerfile builds successfully · [§2.3](GRADING_RUBRIC.md#23-dockerfile-1-point)
- [x] Drift monitoring script runs without errors · [§3](GRADING_RUBRIC.md#3-drift-monitoring-3-points)
- [ ] GitHub Actions workflow passes (green checkmark on PR) · [§4.3](GRADING_RUBRIC.md#43-github-actions-1-point)
- [x] All TODO comments in your code are addressed or justified · [§5.1](GRADING_RUBRIC.md#51-code-quality-1-point)
- [ ] PR is open against the course repo `main` branch with title `[Homework] <Your Full Name>`

---

## MLflow Networking: localhost vs Docker

**Local Development**:
- MLflow server: `http://localhost:5000`
- Python code: `mlflow.set_tracking_uri("http://localhost:5000")`
- All components (Python, MLflow) run on your machine

**Docker Deployment**:
- MLflow server external: `http://localhost:5000` (from your machine)
- MLflow server in Docker network: `http://mlflow:5000` (service name, if using docker-compose)
- Docker build (pulling model): Use `http://localhost:5000` or override with `--build-arg`
- Container runtime: Use the URI passed to the build or set as environment variable

**Quick fix**: If Dockerfile build fails with "Connection refused", ensure MLflow is running:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

---

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

**Column Names Don't Match**:
```bash
# Check the actual column names in your CSV:
python -c "import pandas as pd; df = pd.read_csv('data_pipeline/songs.csv'); print(df.columns.tolist())"

# Common issues:
# - Audio feature column names differ from Spotify API (e.g., 'danceability_score' instead of 'danceability')
# - Target column named 'genre_name' or 'music_genre' instead of 'genre'
# - Year column named 'release_year' or 'year_released' instead of 'year'

# Solution: rename columns in process.py before splitting:
# df = df.rename(columns={'danceability_score': 'danceability', ...})
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

---

## Notes for Students

1. **Genre Classification + Drift**: The task is to classify song **genre** (target) from audio features. The release **year** is NOT the target — it's used to create a temporal train/test split that simulates real data drift.

   **The 2010 boundary marks the Streaming Era Shift:**
   - **Pre-2010 (Training)**: CD/iTunes era — longer songs, more acoustic, higher emotional valence
   - **Post-2010 (Production)**: Spotify/Apple Music era — shorter, punchier, more electronic, heavily compressed

   **Audio features show statistically significant drift:**
   - Loudness: +1.56 dB (loudness wars & compression)
   - Acousticness: -5.75% (more synth, less acoustic)
   - Valence: -6.5% (moodier music)
   - Energy: +4.3% (more intense production)
   - Duration: -8.4 sec (streaming optimization)

2. **Audio Features**: Spotify audio features represent objective measurements of the audio:
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

---

## Grading Rubric

Full point breakdown is in [GRADING_RUBRIC](GRADING_RUBRIC.md) (20 points total).

| Component | Points |
|-----------|--------|
| [Data Pipeline](GRADING_RUBRIC.md#1-data-pipeline-6-points) | 6 |
| [Model Serving](GRADING_RUBRIC.md#2-model-serving-5-points) | 5 |
| [Drift Monitoring](GRADING_RUBRIC.md#3-drift-monitoring-3-points) | 3 |
| [Testing & CI/CD](GRADING_RUBRIC.md#4-testing--cicd-4-points) | 4 |
| [Documentation](GRADING_RUBRIC.md#5-documentation--code-quality-2-points) | 2 |
| **TOTAL** | **20** |

---

## Helpful Resources

- [DVC Docs](https://dvc.org/doc)
- [MLflow Docs](https://mlflow.org/docs/latest/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)

Good luck!
