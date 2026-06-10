# Model Serving

FastAPI service for the Spotify genre classifier.

The champion model (`spotify-genre-classifier@champion`) is pulled from
MLflow at Docker build time and served through two endpoints:

- `GET /health` — liveness probe
- `POST /predict` — genre prediction from 12 audio features

## Local development

```bash
uv sync
uv run python -m mlflow models download -m "models:/spotify-genre-classifier@champion" -d ./models --no-directory
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Tests

```bash
uv run pytest tests/
```
