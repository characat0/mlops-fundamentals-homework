FROM python:3.11-slim

WORKDIR /app

# 👉 Dependencias del servicio FastAPI
COPY model_serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 👉 Código fuente
COPY model_serving/app ./app

# 👉 Forzar versión de NumPy compatible
RUN pip install "numpy<2"

# 👉 Copiar el modelo campeón descargado previamente
COPY models ./models

# 👉 Instalar dependencias del modelo
RUN pip install --no-cache-dir -r ./models/model/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

