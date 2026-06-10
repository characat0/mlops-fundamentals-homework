FROM python:3.11-slim
 
WORKDIR /app
 
COPY requirements.txt .
RUN pip install -r requirements.txt
 
# Pull the @champion model from MLflow and bake it into the image.
# The model was registered in evaluate.py with the alias "champion".
# The container becomes self-contained — no live MLflow server needed at runtime.
#ARG MLFLOW_TRACKING_URI=http://localhost:5000
ARG MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
 
#RUN mlflow models download \
#    -m "models:/spotify-genre-classifier@champion" \
#    -d ./models \
#    --no-directory

RUN python -c "\
import shutil; \
import mlflow; \
client = mlflow.tracking.MlflowClient(); \
mv = client.get_model_version_by_alias('spotify-genre-classifier', 'champion'); \
path = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path='.'); \
shutil.copytree(path, './models', dirs_exist_ok=True) \
"


COPY ./app ./app
RUN touch /app/app/__init__.py
 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]