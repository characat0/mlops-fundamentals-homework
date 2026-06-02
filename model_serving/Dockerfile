FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# TODO: Pull the MLflow model with @champion alias and bake it into the image.
# Students should implement this using:
#   ARG MLFLOW_TRACKING_URI=http://mlflow:5000
#   RUN mlflow models download -m models:/champion@champion -d ./models --no-directory
# Then load the model in app/main.py using mlflow.sklearn.load_model()

COPY ./app ./app

# Ensure the app directory is a Python package
RUN touch /app/app/__init__.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
