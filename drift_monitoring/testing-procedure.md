# Testing Procedure para `analyze_drift.py`

## Descripción General

El script `analyze_drift.py` realiza detección de data drift usando el test estadístico Kolmogorov-Smirnov (KS). Compara la distribución de características de audio entre:
- **Modo batch**: datos de entrenamiento vs simulación de producción
- **Modo online**: datos de entrenamiento vs logs de API en vivo

---

## Pre-requisitos

### 1. **Dependencias del Sistema**
- Python 3.8 o superior
- pip (gestor de paquetes Python)

### 2. **Dependencias de Python**

Instala los siguientes paquetes:

```bash
pip install pandas scipy
```

**Explicación**:
- `pandas`: Lectura y manipulación de archivos CSV y JSONL
- `scipy`: Implementación del test Kolmogorov-Smirnov (scipy.stats)

### 3. **Estructura de Directorios**

Asegúrate de tener la siguiente estructura:

```
mlops-fundamentals-homework/
├── drift_monitoring/
│   ├── src/
│   │   └── analyze_drift.py
│   ├── logs/
│   │   └── api_requests.jsonl (solo para modo online)
│   └── drift_report.json (se genera aquí)
├── data_pipeline/
│   └── data/
│       ├── train.csv
│       └── prod_sim.csv
```

### 4. **Archivos de Datos Requeridos**

#### Para Modo Batch:
- `data_pipeline/data/train.csv`: DataFrame con características de audio del conjunto de entrenamiento
- `data_pipeline/data/prod_sim.csv`: DataFrame con características de audio simuladas de producción

#### Para Modo Online:
- `data_pipeline/data/train.csv`: DataFrame con características de audio del conjunto de entrenamiento
- `drift_monitoring/logs/api_requests.jsonl`: Archivo de logs con formato JSONL (una línea JSON por solicitud)

**Formato esperado de las características**:
```
danceability, energy, key, loudness, mode, speechiness, acousticness, 
instrumentalness, liveness, valence, tempo, duration_ms
```

---

## Procedimiento de Testing

### **Opción 1: Modo Batch**

Este modo compara dos archivos CSV directamente (útil para análisis post-mortem).

#### Pasos:

1. **Preparar los archivos de datos**:
   - Asegúrate de tener `data/train.csv` y `data/prod_sim.csv`
   - Ambos deben contener las mismas columnas de características

2. **Ejecutar el script**:

   ```bash
   uv run python drift_monitoring/src/analyze_drift.py --mode batch --train_data data_pipeline\data\train.csv --prod_data data_pipeline/data/prod_sim.csv --output drift_monitoring/drift_report.json
   ```

3. **Verificar los resultados**:

   ```bash
   cat outputs/drift_report.json
   ```

   **Ejemplo de salida**:
   ```json
   {
     "timestamp": "2024-01-15T10:30:45.123456",
     "train_samples": 8000,
     "production_samples": 2000,
     "features_with_drift": 5,
     "drifted_features": ["energy", "loudness", "tempo", "valence", "acousticness"],
     "drift_percentage": 41.67,
     "status": "DRIFT_DETECTED",
     "details": {
       "energy": {
         "ks_statistic": 0.245,
         "p_value": 0.001,
         "drift_detected": true,
         "train_mean": 0.62,
         "prod_mean": 0.58
       }
     }
   }
   ```

---

### **Opción 2: Modo Online**

Este modo compara datos de entrenamiento contra logs de API en vivo (útil para monitoreo continuo).

#### Pasos:

1. **Requisito previo**: 
   - Asegúrate de que la API esté corriendo y generando logs
   - El middleware debe escribir logs en `drift_monitoring/logs/api_requests.jsonl`

2. **Generar logs (si no existen)**:

   Si la API no está generando logs automáticamente:
   
   ```bash
   # Inicia la API en otra terminal:
   uv run python drift_monitoring/src/analyze_drift.py
   
   # Realiza algunas solicitudes de predicción contra la API
   # Los logs se escribirán automáticamente en drift_monitoring/logs/api_requests.jsonl
   ```

3. **Ejecutar el script**:

   ```bash
   uv run python drift_monitoring/src/analyze_drift.py --mode online --train_data data_pipeline/data/train.csv --api_logs drift_monitoring/logs/api_requests.jsonl --output drift_monitoring/drift_report_online.json
   ```

4. **Verificar los resultados**:

   ```bash
   cat drift_monitoring/drift_report_online.json
   ```

---

## Interpretación de Resultados

### Campos Principales:

| Campo | Descripción |
|-------|-------------|
| `status` | **DRIFT_DETECTED** o **NORMAL** (basado en umbral del 20%) |
| `drift_percentage` | % de características con drift detectado |
| `features_with_drift` | Número absoluto de características con drift |
| `drifted_features` | Array con nombres de características driftadas |

### Interpretación del Test KS:

- **p_value < 0.05**: Drift detectado (rechazo de hipótesis nula)
- **p_value ≥ 0.05**: Sin drift (distribuciones similares)
- **ks_statistic**: Magnitud máxima de la diferencia entre distribuciones

### Umbral de Alerta:

- Si **drift_percentage > 20%**: Status = `DRIFT_DETECTED` 🚨
- Si **drift_percentage ≤ 20%**: Status = `NORMAL` ✅

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'pandas'`

**Solución**:
```bash
pip install pandas scipy
```

### Error: `FileNotFoundError: [Errno 2] No such file or directory`

**Causas posibles**:
- Ruta de archivo incorrecta
- Directorio no existe

**Solución**:
```bash
# Verifica que existan los archivos
ls -la data_pipeline/data/train.csv
ls -la data_pipeline/data/prod_sim.csv

# O en Windows:
dir data_pipeline\data\train.csv
dir data_pipeline\data\prod_sim.csv
```

### Error: `API logs not found`

**Solución**:
- Asegúrate de que la API esté corriendo
- Realiza algunas solicitudes de predicción
- Verifica que el archivo `drift_monitoring/logs/api_requests.jsonl` exista

### Advertencia: `API logs are empty`

**Solución**:
```bash
# Haz predicciones contra la API para generar logs
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"danceability": 0.7, "energy": 0.8, ...}'
```

### Error: `command not found: uv`

**Solución**:
- Instala `uv` (gestor de proyectos Python moderno)
- En Windows: `pip install uv`
- En macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## Ejemplo Completo de Testing

### Setup Inicial:

```bash
# 1. Instalar dependencias
pip install pandas scipy

# 2. Crear directorios si no existen
mkdir -p drift_monitoring/logs
mkdir -p data_pipeline/data
```

### Test Batch:

```bash
uv run python drift_monitoring/src/analyze_drift.py \
  --mode batch \
  --train_data data_pipeline/data/train.csv \
  --prod_data data_pipeline/data/prod_sim.csv \
  --output drift_monitoring/drift_report.json

# Verificar resultado
cat drift_monitoring/drift_report.json | python -m json.tool
```

### Test Online:

```bash
# Terminal 1: Iniciar la API
uv run python drift_monitoring/src/main.py

# Terminal 2: Ejecutar análisis
uv run python drift_monitoring/src/analyze_drift.py \
  --mode online \
  --train_data data_pipeline/data/train.csv \
  --api_logs drift_monitoring/logs/api_requests.jsonl \
  --output drift_monitoring/drift_report.json

# Verificar resultado
cat drift_monitoring/drift_report.json | python -m json.tool
```

---

## Notas Importantes

✅ El script es idempotente: ejecutarlo múltiples veces produce el mismo resultado
✅ El reporte se sobrescribe cada ejecución (guarda versiones anteriores si lo necesitas)
✅ Los logs contienen información detallada: revisa la salida de consola para detalles
✅ El test KS funciona mejor con >30 muestras por grupo (verificado automáticamente)

---

## Contacto

Para preguntas o issues, consulta el archivo README principal del proyecto.
