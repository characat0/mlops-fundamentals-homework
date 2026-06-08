FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pull the @champion model from MLflow and bake it into the image so the
# container is self-contained (no live MLflow server needed at runtime).
# Override the tracking URI per environment with:
#   docker build --build-arg MLFLOW_TRACKING_URI=http://mlflow-host:5000 .
ARG MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
RUN mlflow artifacts download \
    -u "models:/spotify-genre-classifier@champion" \
    -d ./models

# The model lands in ./models/model; predict_genre() auto-resolves the path
# and loads it with the loader matching its flavor (sklearn or xgboost).

COPY ./app ./app
RUN touch /app/app/__init__.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
