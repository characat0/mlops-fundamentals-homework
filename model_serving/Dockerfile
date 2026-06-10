# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the Spotify Genre Classifier API.
#
# Uses the official Astral uv image (ghcr.io/astral-sh/uv) which
# includes Python 3.14 + uv pre-installed. No pip needed.
#
# Model loading strategy (Approach B):
#   models/local/  — champion committed to git (offline fallback)
#   models/remote/ — downloaded from MLflow at build-time (preferred)
#
# Build without MLflow (offline, uses local fallback):
#   docker build -t spotify-api .
#
# Build with MLflow (downloads @champion):
#   docker build --build-arg MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
#                --build-arg DOWNLOAD_MODEL=true -t spotify-api .
#
# Run:
#   docker run --rm -p 8000:8000 spotify-api
# ---------------------------------------------------------------------------

# ---------- Stage 1: builder ----------
FROM ghcr.io/astral-sh/uv:0.11.2-python3.14-trixie-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Copy dependency manifests first for Docker layer caching.
COPY pyproject.toml uv.lock ./

# Install runtime deps (no dev deps, no project wheel).
RUN uv sync --no-dev --no-install-project

# Copy the application code.
COPY app ./app

# --- Model artifacts ---

# Always copy the local fallback (committed to git).
COPY models/local/ /app/models/local/

# Attempt to download @champion from MLflow Model Registry.
# If DOWNLOAD_MODEL=true and MLflow is reachable, the model is saved
# to models/remote/. If it fails, the build continues (fallback works).
ARG MLFLOW_TRACKING_URI=http://localhost:5000
ARG DOWNLOAD_MODEL=false

RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
      uv pip install --system mlflow==3.13.0 && \
      mkdir -p /app/models/remote && \
      mlflow models download \
        --tracking-uri "$MLFLOW_TRACKING_URI" \
        -m "models:/spotify-genre-classifier@champion" \
        -d /app/models/remote \
        --no-directory && \
      echo "✓ Downloaded @champion from MLflow" || \
      echo "⚠ MLflow download failed — will use local fallback"; \
    else \
      echo "⊘ DOWNLOAD_MODEL=false — skipping MLflow download"; \
    fi


# ---------- Stage 2: runtime ----------
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_ENV_MANAGER_REQUIREMENTS_CHECK=true

WORKDIR /app

# Non-root user for security.
RUN groupadd --system --gid 1000 appuser && \
    useradd  --system --uid 1000 --gid appuser \
             --home-dir /app --shell /usr/sbin/nologin appuser

# Copy virtualenv from builder.
COPY --from=builder /app/.venv /app/.venv

# Copy application code.
COPY --from=builder /app/app /app/app

# Copy model artifacts (local fallback always present).
COPY --from=builder /app/models /app/models

# Create logs directory for the request-logging middleware.
# Only chown app code and models (small). Skip .venv — it's read-only
# and owned by root, but accessible by all (755 from the base image).
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app/app /app/models /app/logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
