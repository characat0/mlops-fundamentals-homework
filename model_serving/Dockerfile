FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Descargar el modelo champion desde MLflow y hornearlo en la imagen
ARG MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
RUN mlflow artifacts download -u "models:/spotify-genre-classifier@champion" -d ./models

COPY ./app ./app
RUN touch /app/app/__init__.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
