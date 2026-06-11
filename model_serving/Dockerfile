FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ARG MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

# Descarga el modelo champion al build de Docker.
# Debe existir en MLflow: spotify-genre-classifier@champion
RUN python -c "import mlflow; mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}'); mlflow.artifacts.download_artifacts(artifact_uri='models:/spotify-genre-classifier@champion', dst_path='/app/models')"

COPY ./app ./app

RUN touch /app/app/__init__.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
