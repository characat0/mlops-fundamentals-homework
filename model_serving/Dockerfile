FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# TODO: Pull the @champion model from MLflow and bake it into the image.
#
# The model was registered in evaluate.py with the alias "champion".
# Use the mlflow CLI to download it here so the container is self-contained
# and doesn't need a live MLflow server at runtime.
#
# Hint:
#   ARG MLFLOW_TRACKING_URI=http://localhost:5000
#   RUN mlflow models download -m "models:/spotify-genre-classifier@champion" \
#       -d ./models --no-directory
#
# After this step, load the model in predict_genre() using:
#   mlflow.sklearn.load_model("./models")
ARG MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MODEL_PATH=./models/model
RUN python -c "import mlflow, mlflow.artifacts; mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}'); mlflow.artifacts.download_artifacts(artifact_uri='models:/spotify-genre-classifier@champion', dst_path='./models')"

COPY ./app ./app
RUN touch /app/app/__init__.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
