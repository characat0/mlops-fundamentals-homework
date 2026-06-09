FROM python:3.12-slim

WORKDIR /app

# 1. Instalar dependencias
COPY model_serving/requirements.txt .
RUN pip install -r requirements.txt

# 2. PUENTE DE DATOS: Copiamos los experimentos locales al contenedor temporalmente
COPY data_pipeline/mlruns /app/mlruns_tmp

# 3. Configuración de MLflow (Solo para uso en tiempo de ejecución si es necesario)
ARG MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# 4. EXTRACCIÓN DIRECTA: Localizamos el modelo dentro de mlruns y lo preparamos
# Buscamos el archivo MLmodel y copiamos su carpeta contenedora directamente a ./models
RUN echo "--- LOCALIZANDO MODELO EN MLRUNS ---" && \
    mkdir -p ./models && \
    # Buscamos la carpeta que contiene MLmodel y copiamos su contenido
    find /app/mlruns_tmp -name "MLmodel" -exec dirname {} \; | xargs -I % cp -rv %/. ./models/ && \
    echo "--- VERIFICACIÓN DE /APP/MODELS ---" && \
    ls -la ./models && \
    # Si no se encuentra el modelo, el build fallará aquí
    if [ ! -f ./models/MLmodel ]; then echo "❌ ERROR: No se encontró el modelo en la carpeta mlruns"; exit 1; fi && \
    # Limpiamos los archivos temporales para reducir el tamaño de la imagen
    rm -rf /app/mlruns_tmp

# 5. Copiar artefactos joblib (encoder/scaler) y el código de la API
COPY data_pipeline/models/*.joblib ./models/
COPY model_serving/app ./app
RUN touch /app/app/__init__.py

# 6. Exponer puerto y comando de inicio
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
