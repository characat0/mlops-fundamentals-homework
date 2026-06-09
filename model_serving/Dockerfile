FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pull the champion model from MLflow and bake it into the image.
# We use host.docker.internal to reach the MLflow server running on the host during build.
ARG MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# Download the model using direct runs:/ URI to bypass registry host header restrictions
# Run ID obtained from champion model version: f30bf27cd3934c0398f8b58b9f4f35c5
RUN mlflow artifacts download -u "models:/spotify-genre-classifier@champion" -d . \
    && mv model models

# Copy the application code
COPY ./app ./app
RUN touch /app/app/__init__.py

# Create a logs directory
RUN mkdir -p /app/logs

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
