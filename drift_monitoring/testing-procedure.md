# Testing Procedure para `analyze_drift.py`

## Descripción General

El script `drift_monitoring/src/analyze_drift.py` realiza detección de data drift usando el test estadístico Kolmogorov-Smirnov (KS). Compara la distribución de características de audio entre:
- **Modo batch**: `data_pipeline/data/train.csv` vs `data_pipeline/data/prod_sim.csv`
- **Modo online**: `data_pipeline/data/train.csv` vs `model_serving/logs/api_requests.jsonl`

---

## 1. Pre-requisitos

### Dependencias
Asegúrate de estar en el entorno virtual activo:
```bash
.venv\Scripts\activate
uv sync
```

### Estructura de Directorios
El script asume la estructura del monorepo:
```
mlops-fundamentals-homework/
├── data_pipeline/
│   └── data/
│       ├── train.csv
│       └── prod_sim.csv
├── model_serving/
│   └── logs/
│       └── api_requests.jsonl
└── drift_monitoring/
    └── src/
        └── analyze_drift.py
```

---

## 2. Procedimiento de Testing

### **Opción 1: Modo Batch**

Compara `data/train.csv` vs `data/prod_sim.csv`.

#### Pasos:

1. **Ejecutar el script**:
   ```bash
   cd drift_monitoring
   uv run python src/analyze_drift.py `
     --mode batch `
     --train_data ../data_pipeline/data/train.csv `
     --prod_data ../data_pipeline/data/prod_sim.csv `
     --output drift_report.json
   ```

2. **Verificar los resultados**:
   ```bash
   cat drift_report.json
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

---

### **Opción 2: Modo Online**

Compara `data/train.csv` vs `model_serving/logs/api_requests.jsonl`.

#### Pasos:

1. **Requisito previo**:
   Asegúrate de haber realizado peticiones POST a la API (ver `model_serving/testing-procedure.md`), lo cual genera los logs en `model_serving/logs/api_requests.jsonl`.

2. **Ejecutar el script**:
   ```bash
   cd drift_monitoring
   uv run python src/analyze_drift.py `
     --mode online `
     --train_data ../data_pipeline/data/train.csv `
     --api_logs ../model_serving/logs/api_requests.jsonl `
     --output drift_report_online.json
   ```

3. **Verificar los resultados**:
   ```bash
   cat drift_report_online.json
   ```

---

## 3. Interpretación de Resultados

| Campo | Descripción |
|-------|-------------|
| `status` | **DRIFT_DETECTED** o **NORMAL** (basado en umbral del 20%) |
| `drift_percentage` | % de características con drift detectado |
| `drifted_features` | Lista de nombres de características con drift |

### Test KS:
- **p_value < 0.05**: Drift detectado.
- **p_value ≥ 0.05**: Sin drift.

---

## 4. Troubleshooting

### Error: `FileNotFoundError`
Verifica que las rutas relativas sean correctas desde donde ejecutas el script. Si ejecutas desde la raíz, ajusta las rutas en consecuencia. Los ejemplos anteriores asumen que ejecutas los comandos desde dentro de la carpeta `drift_monitoring/`.

### Los logs de API están vacíos
Asegúrate de que la API esté corriendo y que hayas realizado peticiones POST válidas. Verifica que el archivo `model_serving/logs/api_requests.jsonl` contenga líneas de JSON.
