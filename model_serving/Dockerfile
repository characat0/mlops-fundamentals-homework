FROM python:3.11-slim

WORKDIR /app

# ARG para MLflow URI
ARG MLFLOW_TRACKING_URI=http://localhost:5000

COPY requirements.txt .
RUN pip install -r requirements.txt

# DESCARGAR MODELO durante build
RUN mlflow models download \
    --model-uri "models:/spotify-genre-classifier@champion" \
    --dst-path ./models

COPY ./app ./app
RUN touch /app/app/__init__.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]