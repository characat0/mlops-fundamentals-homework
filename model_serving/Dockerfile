# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the Spotify Genre Classifier API.
#
# Stage 1 (builder):
#   - Installs uv and the full set of runtime dependencies.
#   - Copies the application code and the pre-baked model artifacts
#     (models_local/ + preprocessors/).
#
# Stage 2 (runtime):
#   - Slim Python image, no uv, no MLflow CLI, no compilers.
#   - Copies the .venv and the model artifacts from the builder.
#   - Runs uvicorn as a non-root user.
#
# The @champion model is committed to the repo under models_local/ and
# the preprocessor artifacts live in preprocessors/. This makes the
# build fully reproducible and offline — no MLflow server connection is
# required. To promote a new champion: copy the new model files into
# models_local/, the new preprocessors into preprocessors/, commit,
# rebuild.
#
# Example:
#   docker build -t spotify-api .
#   docker run --rm -p 8000:8000 spotify-api
# ---------------------------------------------------------------------------

# ---------- Stage 1: builder ----------
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install uv once and pin it so the lockfile resolution is reproducible.
RUN pip install --no-cache-dir uv==0.11.2

# Copy only the dependency manifests first to leverage Docker layer caching:
# code changes won't invalidate the (slow) dependency installation layer.
COPY pyproject.toml uv.lock ./

# Install runtime deps into /app/.venv (no dev deps, no project itself).
# `uv sync --no-dev --no-install-project` is the recommended pattern from
# the uv docs for container builds because it skips the project wheel
# (we never `import model_serving`, only `app.main`).
RUN uv sync --no-dev --no-install-project

# Copy the application code (needed for the project to be importable).
COPY app ./app

# Bake the @champion model and the preprocessor artifacts into the
# image. The champion was promoted to the repo under models_local/ and
# the preprocessors live in preprocessors/ (both versioned alongside
# the code, so the image is fully reproducible and self-contained
# at build time — no MLflow server required).
#
# The API loads from /app/models/ via mlflow.pyfunc.load_model and
# joblib.load, so the destination needs MLmodel + model.ubj + the
# three .joblib files all at the same level.
COPY models_local/ /app/models/
COPY preprocessors/*.joblib /app/models/


# ---------- Stage 2: runtime ----------
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app

WORKDIR /app

# Create a non-root user with a stable uid. Using uid 1000 keeps the
# home directory and any bind-mounted volumes writable without surprises.
RUN groupadd --system --gid 1000 appuser && \
    useradd  --system --uid 1000 --gid appuser \
             --home-dir /app --shell /usr/sbin/nologin appuser

# Copy the prepared virtualenv from the builder. Keeping it intact means
# the runtime image does not need uv, pip, compilers, or any build tools.
COPY --from=builder /app/.venv /app/.venv

# Copy the application code and the baked-in model artifacts.
COPY --from=builder /app/app /app/app
COPY --from=builder /app/models /app/models

# Pre-create the logs directory and hand it to the unprivileged user so
# the request-logging middleware can write to api_requests.jsonl. We
# chown the writable paths explicitly (app/, logs/) instead of the
# whole tree, because recursively chowning .venv over hundreds of
# thousands of files is a noticeable slowdown.
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app/app /app/models /app/logs && \
    chmod -R u+rX,g+rX /app/.venv

USER appuser

EXPOSE 8000

# Container-level health check: hits the /health endpoint every 30s and
# considers the container unhealthy after 3 consecutive failures. Matches
# the /health endpoint implemented in app.main.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# `exec` form so the uvicorn process is PID 1 and receives SIGTERM
# directly from the container runtime (no shell wrapping).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
