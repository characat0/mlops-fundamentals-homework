# syntax=docker/dockerfile:1
# ==========================================
# ETAPA 1: Pipeline & Entrenamiento (Builder)
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalar dependencias del pipeline
COPY data_pipeline/requirements.txt ./data_pipeline/
RUN pip install --no-cache-dir -r data_pipeline/requirements.txt

# Copiar configuración de DVC y código del pipeline (sin los datos pesados)
COPY .dvc/ ./.dvc/
COPY data_pipeline/src/ ./data_pipeline/src/
COPY data_pipeline/dvc.yaml data_pipeline/params.yaml ./data_pipeline/

# Configurar MLflow interno en una ruta absoluta fija
ENV MLFLOW_TRACKING_URI=sqlite:////app/data_pipeline/mlflow.db

# EJECUCIÓN CON BIND MOUNT: Montamos el songs.csv original solo durante el repro.
# Esto evita que el archivo de 2GB se guarde en las capas de la imagen.
WORKDIR /app/data_pipeline
RUN --mount=type=bind,source=data_pipeline/songs.csv,target=/app/data_pipeline/songs.csv \
    dvc repro

# ==========================================
# ETAPA 2: Descarga & Preparación (Downloader)
# ==========================================
FROM python:3.12-slim AS downloader

WORKDIR /app
RUN pip install --no-cache-dir mlflow joblib

# MANTENER ESTRUCTURA: Copiamos a la misma ruta exacta (/app/data_pipeline)
# Esto es vital para que las rutas absolutas guardadas en la DB de MLflow sigan siendo válidas
COPY --from=builder /app/data_pipeline /app/data_pipeline

# Usar la misma URI que en el entrenamiento
ENV MLFLOW_TRACKING_URI=sqlite:////app/data_pipeline/mlflow.db

# CUMPLIMIENTO DE RÚBRICA Y FIX DE BUG SILENCIOSO:
# 1. Descargamos el modelo champion (Rúbrica)
# 2. Rescatamos el LabelEncoder (Necesario para predicciones reales)
RUN echo ">>> 1. RESUMEN DEL MODELO GANADOR (metrics.json):" && \
    cat /app/data_pipeline/metrics.json && echo "" && \
    \
    echo ">>> 2. EJECUTANDO DESCARGA DESDE MLFLOW (@champion):" && \
    mlflow artifacts download -u "models:/spotify-genre-classifier@champion" -d /tmp/download && \
    \
    echo ">>> 3. PREPARANDO ESTRUCTURA COMPLETA EN /app/serving_models:" && \
    mkdir -p /app/serving_models && \
    if [ -d "/tmp/download/model" ]; then \
        cp -r /tmp/download/model/* /app/serving_models/; \
    else \
        cp -r /tmp/download/* /app/serving_models/; \
    fi && \
    # Copiar los encoders y scalers que MLflow no incluye por defecto en el sabor base
    if ls /app/data_pipeline/models/*.joblib 1> /dev/null 2>&1; then \
        cp /app/data_pipeline/models/*.joblib /app/serving_models/; \
        echo "SUCCESS: Encoders/Scalers copied to serving directory."; \
    else \
        echo "WARNING: No .joblib files found in /app/data_pipeline/models/"; \
    fi && \
    \
    echo ">>> 4. VERIFICACIÓN DE ARCHIVOS PARA PRODUCCIÓN:" && \
    ls -la /app/serving_models

# ==========================================
# ETAPA 3: Imagen Final (Serving)
# ==========================================
FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias de la API
COPY model_serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar ÚNICAMENTE el paquete completo (Modelo + Encoders)
COPY --from=downloader /app/serving_models ./models

# Copiar el código de la API
COPY model_serving/app ./app
RUN touch /app/app/__init__.py
RUN mkdir -p /app/logs

# Verificación final de la estructura
RUN echo ">>> FINAL: Estructura de /app/models lista para servir con encoders:" && \
    ls -la /app/models

EXPOSE 8000
# El modelo y encoders ya están horneados en /app/models
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
