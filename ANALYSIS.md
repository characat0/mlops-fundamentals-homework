# ANALYSIS.md — Estado del Proyecto y Plan de Trabajo MLOps

> Documento de analisis y hoja de ruta para el homework **"End-to-End MLOps: Spotify Genre Classification & Drift Monitoring"**.
>
> - **Proyecto**: `mlops-fundamentals-homework`
> - **Stack**: DVC · MLflow · FastAPI · Docker · GitHub Actions · (OpenTelemetry como plus futuro)
> - **Puntaje rubrica**: 20 pts
> - **Estado actual**: ~60% provisto, ~40% esqueletos con TODOs

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Dataset](#2-dataset)
3. [Arquitectura del Monorepo](#3-arquitectura-del-monorepo)
4. [Estado Actual vs Requerido](#4-estado-actual-vs-requerido)
5. [Cobertura por Tema de la Rubrica](#5-cobertura-por-tema-de-la-rubrica)
6. [Pipelines CI/CD Adicionales Propuestos](#6-pipelines-cicd-adicionales-propuestos)
7. [Actividades Pendientes (Checklist)](#7-actividades-pendientes-checklist)
8. [Insumos y Prerrequisitos](#8-insumos-y-prerrequisitos)
9. [Orden de Ejecucion Sugerido](#9-orden-de-ejecucion-sugerido)
10. [Riesgos y Pitfalls](#10-riesgos-y-pitfalls)
11. [OpenTelemetry — Guia de Implementacion Futura](#11-opentelemetry--guia-de-implementacion-futura)
12. [Despliegue: Local vs Cloud (AWS)](#12-despliegue-local-vs-cloud-aws)
13. [Portfolio en AWS: Analisis de Costos y Arquitectura](#13-portfolio-en-aws-analisis-de-costos-y-arquitectura)
12. [Anexo A — Snippets YAML de Pipelines CI/CD](#anexo-a--snippets-yaml-de-pipelines-cicd)
13. [Anexo B — Variables de Entorno y Secretos](#anexo-b--variables-de-entorno-y-secretos)
14. [Anexo C — Glosario MLOps](#anexo-c--glosario-mlops)
15. [Anexo D — Comandos Utiles Consolidados](#anexo-d--comandos-utiles-consolidados)
16. [Anexo E — Mapeo Punto-a-Punto con la Rubrica (20 pts)](#anexo-e--mapeo-punto-a-punto-con-la-rubrica-20-pts)
17. [Anexo F — Submission Checklist para el PR](#anexo-f--submission-checklist-para-el-pr)

---

## 1. Resumen Ejecutivo

El proyecto implementa un pipeline MLOps completo para **clasificar el genero musical** (10 clases) de canciones de Spotify a partir de 12 *audio features*. El nucleo del ejercicio es **detectar data/concept drift** simulando un cambio de distribucion: el dataset se divide temporalmente en `year <= 2010` (entrenamiento, era CD/iTunes) y `year > 2010` (produccion simulada, era Spotify/streaming). El modelo entrenado se registra en MLflow, se le asigna el alias `@champion`, y se sirve en una API FastAPI auto-contenida en Docker, que a su vez genera logs que alimentan un segundo modo de deteccion de drift (*online drift*) contra el trafico real.

**Empaquetado / despliegue final**: imagen Docker self-contained que descarga el modelo `@champion` desde MLflow en build-time y lo expone como API REST (`POST /predict`, `GET /health`) en el puerto 8000.

---

## 2. Dataset

| Aspecto | Detalle |
|---|---|
| **Nombre** | 550k Spotify Songs (Audio, Lyrics and Genres) |
| **Fuente** | Kaggle — `serkantysz/550k-spotify-songs-audio-lyrics-and-genres` |
| **Tamano** | ~550 000 filas, ~2-3 GB en disco |
| **MD5 esperado** | `0e71e2c46244acac485bd8c245aa6e56` (registrado en `songs.csv.dvc`) |
| **Target** | `genre` (10 clases) |
| **Features (12)** | `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms` |
| **Metadata (opcional)** | `id`, `name`, `album_name`, `artists`, `lyrics`, `popularity`, `total_artist_followers`, `avg_artist_popularity`, `artist_ids`, `niche_genres` |
| **Columna auxiliar** | `year` — usada SOLO para el split temporal, NO es target |

### Split temporal (nucleo del experimento)

| Split | Regla | Salida | Proposito |
|---|---|---|---|
| Train | `year <= 2010` | `data/train.csv` | Entrenamiento (era CD/iTunes) |
| Prod sim | `year > 2010` | `data/prod_sim.csv` | Deteccion de drift batch (era Spotify/streaming) |

**Drift esperado** (documentado en README):
- Loudness: +1.56 dB (loudness wars & compression)
- Acousticness: -5.75% (mas synth, menos acustico)
- Valence: -6.5% (musica mas melancolica)
- Energy: +4.3% (produccion mas intensa)
- Duration: -8.4 s (optimizacion para streaming)

---

## 3. Arquitectura del Monorepo

```text
mlops-fundamentals-homework/
+-- .github/workflows/ci.yml        # CI: flake8 + pytest (lint + tests)
+-- data_pipeline/
|   +-- dvc.yaml                    # Pipeline: load -> process -> train -> evaluate
|   +-- params.yaml                 # Hiperparametros (LR + XGBoost)
|   +-- songs.csv.dvc               # Hash MD5 del dataset (DVC tracking)
|   +-- src/
|   |   +-- load.py                 # [OK] Carga songs.csv -> data/raw.csv
|   |   +-- process.py              # [TODO] Split temporal train/prod_sim
|   |   +-- train.py                # [TODO] Entrena LR + XGBoost, loguea a MLflow
|   |   +-- evaluate.py             # [TODO] Registra @champion en Model Registry
|   +-- tests/
|       +-- test_load.py            # [OK] 3 tests (existencia, columnas, filas)
|       +-- test_process.py         # [OK] 3 tests (split, boundary 2010, features)
+-- model_serving/
|   +-- app/main.py                 # [TODO] FastAPI: SpotifyFeatures, /health, middleware, predict_genre
|   +-- Dockerfile                  # [TODO] ARG MLFLOW_URI + RUN mlflow models download
|   +-- tests/test_api.py           # [OK] 3 tests (health, valid, invalid payload)
+-- drift_monitoring/
|   +-- src/analyze_drift.py        # [TODO] KS test (batch + online)
|   +-- requirements.txt
+-- .env.example                    # MLFLOW_TRACKING_URI
+-- .flake8                         # max-line-length=100
+-- .gitignore                      # Excluye data/, models/, mlruns/, logs/
+-- README.md
+-- GRADING_RUBRIC.md
```

---

## 4. Estado Actual vs Requerido

### Leyenda
- [OK] = provisto y funcional
- [TODO] = esqueleto con comentarios a implementar
- [PARCIAL] = parcialmente completo

### Inventario por archivo (mapeado a la rubrica)

| Archivo | Estado | Rubrica | Pts | Que falta / Criterio de aceptacion |
|---|---|---|---|---|
| `data_pipeline/songs.csv.dvc` | [OK] | 1.1 | 0.5 | Hash MD5 `0e71e2c46244acac485bd8c245aa6e56` registrado. `dvc status songs.csv.dvc` debe estar limpio |
| `data_pipeline/src/load.py:10` | [OK] | 1.1 | 0.5 | Lee `songs.csv`, escribe `data/raw.csv` con todas las columnas. Verificar con `dvc repro` que produce `data/raw.csv` con column count esperado |
| `data_pipeline/src/process.py:39` | [TODO] | 1.2 | 1.5 | (a) Split `year <= 2010` -> train, `year > 2010` -> prod_sim (boundary INCLUSIVO en train); (b) ambos CSVs producidos; (c) features de audio + `genre` presentes en ambos |
| `data_pipeline/src/train.py:63,71,109` | [TODO] | 1.3 | 2.0 | (a) Carga `data/train.csv`, target=`genre`; (b) entrena **2 modelos** (LR + XGBoost); (c) `mlflow.log_params` y `mlflow.log_metric("accuracy", ...)` por run; (d) runs visibles en MLflow UI con `run_name` y artefactos |
| `data_pipeline/src/evaluate.py:51` | [TODO] | 1.4 | 1.0 | (a) Encuentra el mejor run por accuracy (`metrics.accuracy DESC`); (b) `client.create_model_version(...)` + `client.set_registered_model_alias(name, "champion", version)` |
| `data_pipeline/dvc.yaml:1` | [OK] | 1.5 | 0.5 | 4 stages (load/process/train/evaluate) definidos; `dvc repro` corre end-to-end sin errores |
| `data_pipeline/params.yaml:7` | [OK] | - | - | Hiperparametros basicos (LR: C=1.0, max_iter=1000; XGB: max_depth=6, lr=0.1, n_est=100). Tunear es opcional |
| `data_pipeline/tests/test_load.py:7` | [OK] | 4.1 | 1.0 | 3 tests: existencia, columnas, filas. **Deben pasar** |
| `data_pipeline/tests/test_process.py:8` | [OK] | 4.1 | 1.0 | 3 tests: split, boundary 2010 (year==2010 -> train), presencia de audio features. **Deben pasar** |
| `model_serving/app/main.py:25` | [TODO] | 2.2 | 1.0 | `SpotifyFeatures` Pydantic con 12 campos exactos y tipos (danceability/energy/loudness/etc. -> float, key/mode/duration_ms -> int, ver §2.2) |
| `model_serving/app/main.py:34` | [TODO] | 2.1 | 1.0 | Middleware `log_requests` que escribe a `logs/api_requests.jsonl` con timestamp; reconstruye el Request |
| `model_serving/app/main.py:58` | [TODO] | 2.1 | 1.0 | `GET /health` que retorna `{"status": "healthy"}` con 200 |
| `model_serving/app/main.py:63-74` | [TODO] | 2.1 | 1.0 | `POST /predict` que carga `@champion` de `./models`, hace inference, retorna `PredictionResponse(genre, confidence)` |
| `model_serving/Dockerfile:8` | [TODO] | 2.3 | 1.0 | (a) `ARG MLFLOW_TRACKING_URI`; (b) `RUN mlflow models download -m models:/spotify-genre-classifier@champion -d ./models` |
| `model_serving/tests/test_api.py:8` | [OK] | 4.1 | 1.0 | 3 tests: health, valid payload (200 con genre+confidence), invalid payload (422). **Deben pasar** |
| `drift_monitoring/src/analyze_drift.py:47` | [TODO] | 3.1+3.2 | 3.0 | (a) `run_ks_analysis`: loop por feature con `scipy.stats.ks_2samp`; (b) poblar `details[feature] = {ks_statistic, p_value, drift_detected, train_mean, prod_mean}`; (c) `drifted_features` y `features_with_drift`; (d) modo `batch` (CSVs) y `online` (JSONL) reusan la misma logica |
| `.github/workflows/ci.yml:1` | [OK] | 4.3 | 1.0 | flake8 + pytest data_pipeline/tests + pytest model_serving/tests en cada PR a main |
| `.flake8:1` | [OK] | 4.2 | 1.0 | max-line-length=100, ignore E203. Sin errores flake8 |
| `README.md:1` | [OK] | 5.2 | 0.5 | Documentacion completa, instrucciones followable |
| Setup end-to-end | [OK] | 5.2 | 0.5 | download -> process -> train -> evaluate corre sin errores |
| Sin TODOs pendientes | [TODO] | 5.1 | 1.0 | Todos los `TODO comments` en el codigo deben estar direccionados o justificados |

**Totales**: 6 (data) + 5 (serving) + 3 (drift) + 4 (testing/CI) + 2 (docs) = **20 puntos** distribuidos.

**Bonus/Advanced (no puntaje base, mencionado en README)**: 2 pts posibles por features avanzadas (ver seccion 6 para pipelines plus).

---

## 5. Cobertura por Tema de la Rubrica

### 5.1 DVC — Versionado de Data

**Que hace**:
- `songs.csv.dvc` registra el MD5 esperado del dataset (contrato de integridad).
- `dvc.yaml` define la DAG de 4 stages: `load -> process -> train -> evaluate`.
- `dvc repro` ejecuta solo los stages cuyo input (deps/params/outs) cambio.

**Comandos clave**:
```bash
cd data_pipeline
dvc status songs.csv.dvc   # Verifica que el hash coincide
dvc repro                  # Ejecuta el pipeline completo
dvc dag                    # Visualiza la DAG
dvc status                 # Que stages necesitan re-ejecucion
```

**Verificacion de integridad** (rubrica 1.1):
```bash
# Linux
md5sum songs.csv
# Windows (PowerShell)
Get-FileHash songs.csv -Algorithm MD5
# Comparar con: 0e71e2c46244acac485bd8c245aa6e56
```

### 5.2 MLflow — Experimentos y Artefactos

**Que registrar por run** (en `train.py`):
- Hiperparametros: `mlflow.log_params(model_params)` (C, max_iter, max_depth, learning_rate, n_estimators).
- Metricas: `mlflow.log_metric("accuracy", acc)` y opcionalmente precision/recall/F1.
- Artefactos: `mlflow.sklearn.log_model(model, "model")` o `mlflow.xgboost.log_model(model, "model")`.
- `run_name=model_name` para identificarlo en la UI.

**Model Registry** (en `evaluate.py`):
- `client.create_model_version(name="spotify-genre-classifier", source=model_uri, run_id=...)` devuelve un objeto con `.version`.
- `client.set_registered_model_alias(name="spotify-genre-classifier", alias="champion", version=v)` asigna el alias movil.

**Integracion con Docker**:
- El `Dockerfile` hace `mlflow models download -m models:/spotify-genre-classifier@champion -d ./models` en build-time.
- `predict_genre()` luego carga con `mlflow.sklearn.load_model("./models")`.
- La imagen queda **self-contained** (no necesita MLflow en runtime).

**Networking** (punto clave):
- Local: `http://localhost:5000`
- Docker build desde host: `http://localhost:5000` o `http://host.docker.internal:5000` (Docker Desktop).
- Docker compose con servicio `mlflow`: `http://mlflow:5000`.

### 5.3 Drift Monitoring — KS Test (Batch + Online)

**Modos soportados** (en `analyze_drift.py`):
| Modo | Compara | Caso de uso |
|---|---|---|
| `batch` | `data/train.csv` vs `data/prod_sim.csv` | Drift entre distribuciones historicas (pre/post 2010) |
| `online` | `data/train.csv` vs `logs/api_requests.jsonl` | Drift entre train y trafico real de la API |

**Algoritmo** (funcion `run_ks_analysis`, compartida):
1. Para cada feature en `AUDIO_FEATURES` (12): extraer valores de `train_df` y `prod_df` (dropna).
2. `scipy.stats.ks_2samp(train_values, prod_values)` -> `(ks_statistic, p_value)`.
3. Drift si `p_value < 0.05`.
4. Poblar `drift_results["details"][feature] = {ks_statistic, p_value, drift_detected, train_mean, prod_mean}`.
5. Si drift: incrementar `features_with_drift` y agregar a `drifted_features`.
6. `status = "DRIFT_DETECTED"` si `drift_percentage > 20%`, sino `"NORMAL"`.

**Schema JSON de salida** (rubrica 3.1):
```json
{
  "timestamp": "2024-...",
  "train_samples": 123456,
  "production_samples": 78901,
  "features_with_drift": 8,
  "drifted_features": ["loudness", "energy", ...],
  "drift_percentage": 66.7,
  "status": "DRIFT_DETECTED",
  "details": {
    "loudness": {
      "ks_statistic": 0.12,
      "p_value": 0.0001,
      "drift_detected": true,
      "train_mean": -8.5,
      "prod_mean": -6.9
    }
  }
}
```

### 5.4 Model Serving — FastAPI + Docker

**Endpoints** (en `main.py`):
| Metodo | Path | Body | Respuesta |
|---|---|---|---|
| `GET` | `/health` | - | `{"status": "healthy"}` (200) |
| `POST` | `/predict` | `SpotifyFeatures` JSON | `PredictionResponse(genre, confidence)` |

**`SpotifyFeatures` Pydantic** (12 campos exactos, tipos importantes):
```python
class SpotifyFeatures(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
```

**Middleware `log_requests`** (en `main.py`):
- Solo loguear `POST /predict`.
- `body_bytes = await request.body()`.
- Parsear JSON, agregar `"timestamp": datetime.utcnow().isoformat()`.
- Append linea JSONL a `logs/api_requests.jsonl`.
- Reconstruir el `Request` con un `receive()` async para que el endpoint pueda releerlo.

**`predict_genre()`** flujo:
1. `model = mlflow.sklearn.load_model("./models")`.
2. Construir `feature_vector` con el orden exacto del entrenamiento.
3. `prediction = model.predict([feature_vector])` y `predict_proba` para confidence.
4. Mapear indice numerico a nombre de genero (usar el `LabelEncoder` guardado o lista hardcodeada).
5. `return PredictionResponse(genre=..., confidence=float(probabilities[0].max()))`.

**Dockerfile** (esqueleto a completar):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG MLFLOW_TRACKING_URI=http://localhost:5000
RUN mlflow models download -m "models:/spotify-genre-classifier@champion" -d ./models --no-directory

COPY ./app ./app
RUN touch /app/app/__init__.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.5 CI/CD — GitHub Actions (workflow actual)

`.github/workflows/ci.yml` ejecuta en cada PR a `main`:
1. `pip install flake8 pytest` + requirements de los 3 modulos.
2. `flake8 .` (lint).
3. `pytest data_pipeline/tests` y `pytest model_serving/tests`.

**Punto clave para la rubrica 4.3**: el PR debe pasar en verde para obtener el punto. Ver Anexo A para pipelines adicionales.

### 5.6 OpenTelemetry — Documentacion para Implementacion Futura

Ver seccion completa en [§11](#11-opentelemetry--guia-de-implementacion-futura).

---

## 6. Pipelines CI/CD Adicionales Propuestos

### Contexto

El workflow actual solo cubre **lint + tests**. En proyectos MLOps reales en ambientes productivos se requieren pipelines adicionales que cierren el ciclo de vida del modelo. Tras revisar patrones usados en plataformas como Netflix (Metaflow), Uber (Michelangelo), Spotify (Klopoki/Lifecycle), Google (Vertex AI Pipelines) y proyectos open-source maduros, identifico los siguientes pipelines de alto valor para este repo:

### Pipelines prioritarios (los 3 pedidos)

#### 6.1 `docker-build.yml` — Build y Publicacion de Imagen

**Proposito**: Garantizar que la imagen Docker compila en cada cambio; publicar versiones tagged a GHCR para despliegues reproducibles.

**Trigger**:
- `push` a `main` (imagen `:dev`).
- `push` de tags `v*.*.*` (imagenes versionadas semanticas).
- `pull_request` (solo build, no push — valida que compila).

**Valor en produccion**:
- Cache de layers (Buildx) reduce tiempo de CI de ~10 min a ~2 min.
- Tag con SHA del commit para trazabilidad bit-a-bit.
- Imagen lista para `docker run` / `docker compose up` / Kubernetes.

#### 6.2 `concept-drift-check.yml` — Deteccion Programada de Drift

**Proposito**: Ejecutar el script de drift periodicamente y en PRs, fallando el build si el drift supera un umbral.

**Trigger**:
- `schedule` (cron semanal: `0 6 * * 1`).
- `workflow_dispatch` (manual desde la UI).
- `pull_request` que modifique `data_pipeline/`, `model_serving/`, o `drift_monitoring/`.

**Valor en produccion**:
- Deteccion temprana de degradacion antes de que impacte a usuarios finales.
- Alimenta la decision de reentrenar (puede disparar `retrain-on-drift.yml`).
- Crea un artefacto `drift_report.json` descargable desde la corrida.

#### 6.3 `dvc-validate.yml` — Validacion de Data y Pipeline

**Proposito**: Asegurar que el dataset y el pipeline siguen siendo reproducibles.

**Trigger**:
- `pull_request` que modifique `songs.csv.dvc`, `dvc.yaml`, `params.yaml`, o archivos en `data_pipeline/src/`.

**Valor en produccion**:
- Evita que entren datasets corruptos (verifica hash MD5 contra `songs.csv.dvc`).
- Garantiza que cambios en hiperparametros o stages del pipeline no rompen el DAG.
- En organizaciones reguladas, provee audit trail de cambios en data de entrenamiento.

### Pipelines adicionales sugeridos (plus)

#### 6.4 `retrain-on-drift.yml` — Reentrenamiento Automatico

**Proposito**: Cuando `concept-drift-check.yml` detecta drift significativo, dispara reentrenamiento y re-evaluacion.

**Trigger**: `workflow_run` del job de drift-check si termina en exito y `status == "DRIFT_DETECTED"`.

**Valor**: Cierra el loop MLOps completo: **drift detectado -> reentrenar -> evaluar -> promover nuevo @champion**.

#### 6.5 `model-promote.yml` — Promocion Controlada de Modelos

**Proposito**: Gates de calidad antes de asignar el alias `@champion` (e.g., accuracy minima, tests pasando, sin regresiones).

**Trigger**: Manual o post-`evaluate.py`.

**Valor**: Implementa el patron **champion/challenger** usado en Netflix y Uber; previene deployment de modelos inferiores.

#### 6.6 `security-scan.yml` — Escaneo de Vulnerabilidades

**Proposito**: Trivy (o Snyk) sobre la imagen Docker y `pip-audit` sobre `requirements.txt`.

**Trigger**: `push` a `main`, `pull_request`, semanal.

**Valor**: CVEs en dependencias son una de las principales causas de incidentes en produccion; este pipeline es estandar en cualquier equipo con SLAs de seguridad.

#### 6.7 `release.yml` — Versionado y Release Notes

**Proposito**: Versionado semantico automatico basado en conventional commits; generacion de release notes; tagging de imagenes Docker.

**Valor**: Trazabilidad de que version del modelo esta en produccion (auditoria, rollback).

### Resumen de pipelines

| Pipeline | Trigger | Valor Principal |
|---|---|---|
| `ci.yml` (existente) | PR | Lint + tests basicos |
| `docker-build.yml` | push/PR | Imagen reproducible, cache, publicacion |
| `concept-drift-check.yml` | cron/PR | Deteccion temprana de degradacion |
| `dvc-validate.yml` | PR | Integridad de data y reproducibilidad |
| `retrain-on-drift.yml` (plus) | workflow_run | Automatizacion del reentrenamiento |
| `model-promote.yml` (plus) | manual/post-eval | Gates de calidad champion/challenger |
| `security-scan.yml` (plus) | push/semanal | CVEs, compliance |
| `release.yml` (plus) | tag | Versionado y trazabilidad |

**Snippets YAML completos en [Anexo A](#anexo-a--snippets-yaml-de-pipelines-cicd).**

---

## 7. Actividades Pendientes (Checklist)

> Cada item referencia el criterio exacto de la rubrica (`§1.1`, `§2.3`, etc.) para asegurar la mayor nota. **Total: 20 puntos**.

### Stage 1 — Data Pipeline (6 pts) → Rubrica §1

- [ ] **§1.1 (0.5 pt)** Descargar `songs.csv` desde Kaggle y colocarlo en `data_pipeline/`
- [ ] **§1.1 (0.5 pt)** Verificar MD5: `0e71e2c46244acac485bd8c245aa6e56` con `md5sum` o `Get-FileHash -Algorithm MD5`
- [ ] **§1.1 (0.5 pt)** Verificar con `dvc status songs.csv.dvc` → debe estar limpio (no "not in cache")
- [ ] **§1.1 (0.5 pt)** `dvc repro` produce `data/raw.csv` con el column count esperado (~17 cols: id, name, album, artists, genre, year, popularity, audio features, etc.)
- [ ] **§1.2 (0.5 pt)** Implementar `process.py`: boolean indexing con `year <= 2010` (INCLUSIVO) y `year > 2010` (EXCLUSIVO)
- [ ] **§1.2 (0.5 pt)** Ambos `data/train.csv` y `data/prod_sim.csv` son producidos con `to_csv(..., index=False)`
- [ ] **§1.2 (0.5 pt)** Audio features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms) + `genre` presentes en ambos CSVs
- [ ] **§1.2** Correr `pytest data_pipeline/tests/test_process.py` → 3 tests pasan
- [ ] **§1.3 (0.5 pt)** `train.py` carga `data/train.csv` con target=`genre`
- [ ] **§1.3 (0.5 pt)** Entrena 2 modelos: LogisticRegression + XGBoost (params desde `params.yaml`)
- [ ] **§1.3 (0.5 pt)** `mlflow.log_params(...)` y `mlflow.log_metric("accuracy", ...)` por cada run
- [ ] **§1.3 (0.5 pt)** Runs visibles en MLflow UI (`http://localhost:5000`) con `run_name` y artefactos (model files)
- [ ] **§1.3** Iniciar MLflow server en otra terminal: `mlflow server --host 0.0.0.0 --port 5000`
- [ ] **§1.3** Correr `dvc repro` y ver 2 runs (uno por modelo) en la UI
- [ ] **§1.4 (0.5 pt)** `evaluate.py` busca runs con `order_by=["metrics.accuracy DESC"]` y selecciona el mejor
- [ ] **§1.4 (0.5 pt)** `client.create_model_version(name="spotify-genre-classifier", source=model_uri, run_id=...)` registra la version
- [ ] **§1.4 (0.5 pt)** `client.set_registered_model_alias(name, "champion", version)` asigna el alias
- [ ] **§1.4** Verificar `metrics.json` generado con `best_run_id`, `best_accuracy`, `model_type`, `model_name`, `champion_alias`
- [ ] **§1.4** Verificar en MLflow UI: `Models -> spotify-genre-classifier -> Aliases -> champion -> vN`
- [ ] **§1.5 (0.5 pt)** `dvc repro` corre end-to-end sin errores (load → process → train → evaluate)

### Stage 2 — Model Serving (5 pts) → Rubrica §2

- [ ] **§2.2 (1 pt)** Definir `SpotifyFeatures` con los 12 campos exactos y tipos:
  - float: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
  - int: key, mode, duration_ms
- [ ] **§2.1 (1 pt)** `GET /health` retorna `{"status": "healthy"}` con 200
- [ ] **§2.1 (1 pt)** `POST /predict` acepta payload `SpotifyFeatures` y retorna `PredictionResponse(genre, confidence)` con 200
- [ ] **§2.1 (1 pt)** Middleware `log_requests` escribe a `logs/api_requests.jsonl` con timestamp en cada `POST /predict`
- [ ] **§2.1** Middleware reconstruye el `Request` (con `async def receive()`) para que el endpoint pueda leer el body
- [ ] **§2.1** `predict_genre()` carga el modelo con `mlflow.sklearn.load_model("./models")`
- [ ] **§2.1** `predict_genre()` extrae las 12 features en el orden correcto, llama `model.predict()` y `model.predict_proba()`
- [ ] **§2.1** Mapea el indice numerico predicho al nombre del genero (lista hardcodeada o LabelEncoder persistido)
- [ ] **§2.1** `predict_genre()` retorna `PredictionResponse(genre=<str>, confidence=<float>)`
- [ ] **§2.1** Correr `pytest model_serving/tests/test_api.py` → 3 tests pasan
- [ ] **§2.1** Probar manualmente con `curl POST /predict` y verificar que se loguea a `logs/api_requests.jsonl`
- [ ] **§2.3 (0.5 pt)** `Dockerfile` tiene `ARG MLFLOW_TRACKING_URI=http://localhost:5000`
- [ ] **§2.3 (0.5 pt)** `Dockerfile` tiene `RUN mlflow models download -m models:/spotify-genre-classifier@champion -d ./models --no-directory`
- [ ] **§2.3** `docker build -t spotify-api:latest ./model_serving` (con MLflow corriendo) → SUCCESS
- [ ] **§2.3** `docker run -p 8000:8000 spotify-api:latest` y probar `POST /predict` desde el host

### Stage 3 — Drift Monitoring (3 pts) → Rubrica §3

- [ ] **§3.1 (0.5 pt)** `run_ks_analysis` carga `data/train.csv` y `data/prod_sim.csv` en modo `batch`
- [ ] **§3.1 (0.5 pt)** Loop por `features_to_test` con `scipy.stats.ks_2samp(train_values, prod_values)` para cada audio feature
- [ ] **§3.1 (0.5 pt)** `drift_results["details"][feature]` contiene `ks_statistic`, `p_value`, `drift_detected`, `train_mean`, `prod_mean`
- [ ] **§3.1** `status` es `"DRIFT_DETECTED"` si `drift_percentage > 20`, sino `"NORMAL"`
- [ ] **§3.1** Features con `p_value < 0.05` se agregan a `drifted_features` y se cuenta en `features_with_drift`
- [ ] **§3.1** `drift_report.json` se escribe con `json.dump(drift_results, f, indent=2)`
- [ ] **§3.1** Correr: `python src/analyze_drift.py --mode batch --train_data ../data_pipeline/data/train.csv --prod_data ../data_pipeline/data/prod_sim.csv --output batch_drift_report.json`
- [ ] **§3.1** Verificar que el JSON tiene el schema esperado (ver §5.3)
- [ ] **§3.2 (0.5 pt)** `analyze_online_drift` carga `data/train.csv` y `logs/api_requests.jsonl`
- [ ] **§3.2 (0.5 pt)** Parsea JSONL linea por linea y construye DataFrame con las features
- [ ] **§3.2 (0.5 pt)** Reusa `run_ks_analysis` (misma funcion para ambos modos)
- [ ] **§3.2** Hacer al menos 1 `POST /predict` para generar logs
- [ ] **§3.2** Correr: `python src/analyze_drift.py --mode online --train_data ... --api_logs ../model_serving/logs/api_requests.jsonl --output online_drift_report.json`

### Stage 4 — Testing & CI/CD (4 pts) → Rubrica §4

- [ ] **§4.1 (1 pt)** `pytest data_pipeline/tests -v` → 6 tests pasan (3 load + 3 process)
- [ ] **§4.1 (1 pt)** `pytest model_serving/tests -v` → 3 tests pasan (health + valid + invalid)
- [ ] **§4.2 (1 pt)** `flake8 .` → 0 errores (warnings OK; el `.flake8` ya tiene `max-line-length=100` y `extend-ignore = E203`)
- [ ] **§4.3 (1 pt)** Abrir PR contra `main`; CI pasa en verde (lint + ambos pytests)
- [ ] **Plus (no obligatorio)** Crear `docker-build.yml` (ver Anexo A.1)
- [ ] **Plus (no obligatorio)** Crear `concept-drift-check.yml` (ver Anexo A.2)
- [ ] **Plus (no obligatorio)** Crear `dvc-validate.yml` (ver Anexo A.3)

### Stage 5 — Documentation & Code Quality (2 pts) → Rubrica §5

- [ ] **§5.1 (1 pt)** Todos los `TODO comments` en el codigo estan direccionados o justificados (buscar con `grep -r "TODO" data_pipeline/ model_serving/ drift_monitoring/`)
- [ ] **§5.1** Codigo sigue convenciones PEP 8 (lo valida flake8)
- [ ] **§5.2 (0.5 pt)** README es claro, instrucciones son followable
- [ ] **§5.2 (0.5 pt)** Setup funciona end-to-end: download Kaggle → process → train → evaluate → API → drift

### Entrega Final

- [ ] Fork del repo, branch `solution/<tu-nombre>`, commits incrementales con conventional commits
- [ ] PR con titulo exacto: `[Homework] <Tu Nombre Completo>`
- [ ] Descripcion del PR incluye el Submission Checklist completado (ver Anexo F)
- [ ] CI en verde en el PR (otorga el punto de §4.3)
- [ ] (Opcional) Agregar los 3 pipelines plus del Anexo A para bonus/advanced

**Auto-verificacion final** (corre estos comandos antes de hacer el PR):
```bash
# Lint
flake8 .

# Tests (ambos modulos)
pytest data_pipeline/tests -v
pytest model_serving/tests -v

# Drift batch (requiere haber corrido process.py)
python drift_monitoring/src/analyze_drift.py \
  --mode batch \
  --train_data data_pipeline/data/train.csv \
  --prod_data data_pipeline/data/prod_sim.csv \
  --output /tmp/drift_check.json
test -f /tmp/drift_check.json && echo "Drift batch OK"

# Buscar TODOs residuales
grep -r "TODO\|FIXME\|XXX" data_pipeline/src model_serving/app drift_monitoring/src || echo "No hay TODOs residuales"

# DVC status
cd data_pipeline && dvc status

---

## 8. Insumos y Prerrequisitos

### Software

| Herramienta | Version | Notas |
|---|---|---|
| Python | 3.9+ (CI usa 3.9; local puede ser 3.11) | `python --version` |
| Git | cualquiera reciente | `git --version` |
| Docker Desktop | ultima estable | Solo para Stage 2.3 |
| DVC | 3.0+ | Se instala via pip |
| MLflow | 2.3+ | `mlflow server --host 0.0.0.0 --port 5000` |
| Kaggle CLI | ultima | `pip install kaggle` |

### Credenciales y Cuentas

- **Cuenta Kaggle** (gratis) — [kaggle.com](https://www.kaggle.com).
- **API token Kaggle**: descargar desde [kaggle.com/settings/account](https://www.kaggle.com/settings/account).
  - Guardar como `~/.kaggle/kaggle.json` (Linux/Mac) o `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
  - En Windows: `mkdir $env:USERPROFILE\.kaggle` y copiar ahi.
- **Cuenta GitHub** — para fork y PR.

### Espacio en Disco

- Dataset: ~500 MB comprimido, ~2-3 GB descomprimido.
- Modelos + MLflow artifacts: ~200-500 MB.
- Imagen Docker: ~1-1.5 GB.
- **Total recomendado**: ~5-6 GB libres.

### Tiempo Estimado

- Setup inicial (Kaggle + deps + DVC repro): 15-20 min.
- Implementacion de TODOs: 2-4 horas (dependiendo de experiencia).
- Debugging de integracion MLflow/Docker: 30-60 min.

### Comandos de Setup

```bash
# 1. Clonar fork
git clone https://github.com/<tu-usuario>/mlops-fundamentals-homework.git
cd mlops-fundamentals-homework
git checkout -b solution/<tu-nombre>

# 2. Crear venv e instalar
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r data_pipeline/requirements.txt
pip install -r model_serving/requirements.txt
pip install -r drift_monitoring/requirements.txt
pip install kaggle

# 3. Configurar Kaggle
#   - Descargar kaggle.json desde https://www.kaggle.com/settings/account
#   - Guardar en ~/.kaggle/kaggle.json (Linux/Mac) o %USERPROFILE%\.kaggle\kaggle.json (Win)

# 4. Descargar dataset
kaggle datasets download -d serkantysz/550k-spotify-songs-audio-lyrics-and-genres
# Mover/descomprimir songs.csv a data_pipeline/

# 5. Configurar env
cp .env.example .env
# Editar .env si tu MLflow server esta en otra URI

# 6. Iniciar MLflow (en otra terminal)
mlflow server --host 0.0.0.0 --port 5000

# 7. Correr pipeline
cd data_pipeline
dvc repro
```

---

## 9. Orden de Ejecucion Sugerido

1. **Setup completo** (seccion 8) — Kaggle, deps, MLflow server.
2. **Stage 1** — `process.py` -> tests pasan -> `train.py` -> ver runs en MLflow UI -> `evaluate.py` -> verificar `@champion` -> `dvc repro` end-to-end.
3. **Stage 2** — `SpotifyFeatures` -> `GET /health` -> tests pasan -> `predict_genre()` -> `uvicorn` local y probar manualmente -> `Dockerfile` -> `docker build` -> `docker run` y probar.
4. **Stage 3** — `analyze_drift.py` --mode batch -> verificar drift significativo -> hacer 5-10 POST /predict -> --mode online -> verificar log parsing.
5. **Stage 4** — `flake8 .` limpio -> `pytest` ambos modulos limpio -> commit -> push -> abrir PR -> CI verde.
6. **Plus (CI/CD avanzado)** — Crear los 3 pipelines del Anexo A, configurar secrets (`GHCR_TOKEN`, `MLFLOW_TRACKING_URI`).
7. **Plus (OpenTelemetry)** — Seguir guia de seccion 11 cuando se requiera instrumentacion real.

---

## 10. Riesgos y Pitfalls

### Pitfalls documentados en el README

| # | Riesgo | Mitigacion |
|---|---|---|
| 1 | **Off-by-one en year split**: `year < 2010` en vez de `<= 2010` | Los tests en `test_process.py:90` validan boundary exacto |
| 2 | **Olvidar `LabelEncoder`**: modelos sklearn necesitan labels numericos | Codificar siempre `genre` antes de fit |
| 3 | **No loguear a MLflow**: dejar metricas solo en prints | Verificar UI en `localhost:5000` despues de cada run |
| 4 | **Mismatch de campos Pydantic**: nombres/tipos no coinciden con tests | `test_api.py:16` define el payload canonico |
| 5 | **Git no trackea archivos**: archivos importantes no commiteados | `git status` antes de push |

### Pitfalls adicionales detectados

| # | Riesgo | Mitigacion |
|---|---|---|
| 6 | **MLflow server no corriendo durante build Docker**: `Connection refused` al hacer `mlflow models download` | Iniciar MLflow ANTES de `docker build`; verificar con `curl http://localhost:5000` |
| 7 | **Columnas renombradas en dataset Kaggle**: el README menciona `danceability_score`, `genre_name`, `release_year` como nombres alternativos | Inspeccionar columnas: `python -c "import pandas as pd; print(pd.read_csv('data_pipeline/songs.csv').columns.tolist())"` y renombrar en `process.py` |
| 8 | **Scaler no guardado**: si en `train.py` escalas para LR pero no guardas el `StandardScaler`, `predict_genre()` en la API no podra escalar las features de entrada | Loguear el scaler como artefacto MLflow o usar un Pipeline de sklearn (`make_pipeline(StandardScaler(), LogisticRegression())`) que MLflow serializa completo |
| 9 | **LabelEncoder no guardado**: el mapeo indice->genero se pierde entre training y serving | Guardar el encoder como artefacto o hardcodear la lista ordenada de generos en la API (los 10 son fijos segun README) |
| 10 | **Path del modelo en Docker**: `mlflow.sklearn.load_model("./models")` asume estructura correcta | Verificar que `mlflow models download --no-directory` deja `MLmodel`, `model.pkl`, etc., directamente en `./models/` |
| 11 | **Windows + Docker networking**: `localhost` en `MLFLOW_TRACKING_URI` puede no resolver desde el contenedor | Usar `host.docker.internal:5000` (Docker Desktop) o exponer MLflow en `0.0.0.0` |
| 12 | **Drift check con dataset sintetico**: si el split no tiene drift, el KS test no detectara nada | El README garantiza drift pre/post 2010; si los tests pasan, el drift estara presente |
| 13 | **CI con Python 3.9 vs local 3.11**: diferencias menores en type hints y deprecations | Mantener compatibilidad 3.9 (CI usa 3.9) |
| 14 | **Flake8 estricto**: lineas >100 chars, imports no usados | Configurar `.flake8` ya provee `max-line-length=100`; ejecutar `flake8 .` antes de push |

---

## 12. Despliegue: Local vs Cloud (AWS)

> **Respuesta corta**: este homework **no requiere cloud** — todo corre en local. AWS es **opcional** y vale para bonus/portfolio. Esta seccion documenta ambas rutas para que decidas.

### 12.1 ¿Que se necesita como minimo (modo local)?

| Componente | Donde corre | Comando |
|---|---|---|
| Python 3.9+ | Tu maquina | `python --version` |
| DVC | Tu maquina | `pip install dvc` |
| MLflow server | Tu maquina | `mlflow server --host 0.0.0.0 --port 5000` |
| FastAPI | Tu maquina o Docker local | `uvicorn app.main:app --port 8000` |
| Docker (opcional, para §2.3) | Docker Desktop | `docker build -t spotify-api ./model_serving` |
| Kaggle CLI | Tu maquina | `pip install kaggle` |
| Git + GitHub | Tu maquina + web | PR contra el repo del curso |

**Costo**: $0. **Espacio en disco**: ~5-6 GB.

### 12.2 Limitaciones del modo local

- MLflow server se cae si cierras la terminal (a menos que uses `nohup` o `tmux`).
- `docker build` requiere MLflow accesible desde el daemon de Docker (con Docker Desktop en Windows/Mac funciona con `host.docker.internal`).
- No hay alta disponibilidad ni auto-scaling.
- DVC cache es local; no hay backup remoto.
- Drift detection batch requiere correr el script manualmente; no hay scheduler.

**Para el homework, ninguna de estas limitaciones afecta los 20 puntos.** La rúbrica evalúa funcionalidad, no disponibilidad.

### 12.3 Roadmap de despliegue en AWS (bonus/portfolio)

Si quieres llevar el proyecto a AWS, hay 4 niveles de madurez:

#### Nivel 1 — Solo storage remoto (facil, +valor)

**Que**: Usar **S3** como DVC remote en vez de cache local. La API y MLflow siguen locales.

**Servicios**: S3 + IAM user con access keys.

**Setup**:
```bash
# 1. Crear bucket S3
aws s3 mb s3://mlops-spotify-dvc --region us-east-1

# 2. Configurar DVC remote
cd data_pipeline
dvc remote add -d myremote s3://mlops-spotify-dvc
dvc push   # Sube data/, models/ a S3
dvc pull   # En otra maquina, descarga
```

**Costo**: ~$0.023/GB/mes (dataset 2.5 GB ≈ $0.06/mes).
**Tiempo**: 30 min.

#### Nivel 2 — MLflow server en EC2 (medio)

**Que**: Correr MLflow server en una instancia EC2 accesible desde internet, con backend store en S3 y artifact store en S3.

**Servicios**: EC2 t2.micro (free tier 12 meses) + S3.

**Setup resumido**:
```bash
# En EC2 (Amazon Linux 2)
sudo yum install python3 -y
pip3 install mlflow boto3
export MLFLOW_S3_BUCKET=mlops-spotify-mlflow
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://mlops-spotify-mlflow/artifacts

# Security group: abrir puerto 5000 al mundo (o tu IP/32)
# En tu maquina:
export MLFLOW_TRACKING_URI=http://<EC2-PUBLIC-IP>:5000
```

**Costo**: t2.micro gratis 12 meses; despues ~$8/mes.
**Tiempo**: 1-2 horas (incluye setup de IAM, security groups, elastic IP opcional).

#### Nivel 3 — API en ECS Fargate (medio-alto)

**Que**: Publicar la imagen Docker `spotify-api` en **Amazon ECR** y correrla en **ECS Fargate** (serverless containers, sin administrar EC2).

**Servicios**: ECR + ECS Fargate + IAM Roles.

**Setup resumido**:
```bash
# 1. Crear repo ECR
aws ecr create-repository --repository-name spotify-api

# 2. Login + push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
docker tag spotify-api:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/spotify-api:latest
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/spotify-api:latest

# 3. Crear task definition + service en ECS
#    - Image: <ECR_URI>
#    - CPU: 256 (.25 vCPU), Memory: 512 MB
#    - Port mapping: 8000
#    - Env var: MLFLOW_TRACKING_URI=http://<EC2>:5000
```

**Costo**: Fargate ~$0.04/hora para 0.25 vCPU + 0.5 GB (~$30/mes 24/7). Free tier incluye ciertos limites el primer mes.
**Tiempo**: 3-4 horas (incluye VPC, subnets, security groups, ALB opcional).

#### Nivel 4 — Full MLOps stack en AWS (avanzado, overkill para homework)

**Que**: Sagemaker Pipelines (orquestacion) + Sagemaker Endpoints (serving) + S3 (DVC) + CloudWatch (monitoreo) + CodePipeline (CI/CD) + Lambda (drift check).

**Servicios**: ~10 servicios AWS.

**Costo**: $50-200/mes. **NO recomendado** para este homework — la rúbrica no lo exige y la complejidad no aporta puntos extra.

### 12.4 Comparativa de opciones AWS

| Opcion | Servicios | Costo/mes | Tiempo setup | Valor para homework |
|---|---|---|---|---|
| Solo S3 DVC remote | S3 | <$1 | 30 min | Bajo (no aporta pts) |
| MLflow en EC2 | EC2 + S3 | $0-8 | 1-2 h | Medio (portfolio) |
| API en ECS Fargate | ECR + ECS + S3 + EC2 | $30-40 | 3-4 h | Alto (portfolio) |
| Full Sagemaker stack | ~10 servicios | $50-200 | 1-2 dias | Nulo (overkill) |

### 12.5 AWS y OpenTelemetry

Si despliegas en AWS, la integracion OTel es natural:

- **AWS Distro for OpenTelemetry (ADOT)**: imagen Docker del OTel Collector optimizada para AWS.
- **X-Ray como backend de tracing** (alternativa a Jaeger/Tempo): `opentelemetry-exporter-aws-xray`.
- **CloudWatch como backend de metrics**: `opentelemetry-exporter-aws-cloudwatch`.
- **CloudWatch Logs** para structured logs (correlacionados con `trace_id`).

**Comando ejemplo** (en el Dockerfile o docker-compose):
```yaml
# docker-compose.yml (solo referencia, no se incluye en el homework)
services:
  api:
    image: spotify-api:latest
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://adot-collector:4317
      - AWS_REGION=us-east-1
  adot-collector:
    image: public.ecr.aws/aws-observability/aws-otel-collector:latest
    command: ["--config=/etc/ecs/ecs-xray.yaml"]
```

### 12.6 Recomendacion final

**Para entregar el homework (20 pts)**:
- Quedate en local. Es lo que evalua la rubrica.
- Si quieres asegurarte de que Docker build funciona, usa Docker Desktop local.

**Para portfolio/plus (opcional)**:
- Nivel 1 (S3) si quieres mostrar que sabes versionado de data en cloud.
- Nivel 2 (EC2 para MLflow) si quieres mostrar tracking de experimentos en cloud.
- Nivel 3 (ECS Fargate) si quieres mostrar model serving en cloud.

**No hagas**:
- Nivel 4 (Sagemaker stack) — pierde tiempo sin puntos.
- Despliegues "completos" con Terraform/CDK — no es el alcance del lab.

### 12.7 Checklist rapido si vas a AWS

```bash
# Pre-requisitos
aws --version          # AWS CLI v2
aws configure          # Set account, region, output

# Nivel 1: S3 DVC remote
aws s3 mb s3://mlops-spotify-dvc
cd data_pipeline && dvc remote add -d myremote s3://mlops-spotify-dvc && dvc push

# Nivel 2: MLflow en EC2
# - Lanzar t2.micro con Amazon Linux 2
# - Security group: inbound 5000 from 0.0.0.0/0 (o tu IP)
# - SSH, instalar python3 + pip + mlflow
# - Correr mlflow server con --default-artifact-root s3://...

# Nivel 3: API en Fargate
aws ecr create-repository --repository-name spotify-api
# (build + push imagen, ver seccion §5.4)
# (crear task definition + service en consola ECS)
```

---

## 11. OpenTelemetry — Guia de Implementacion Futura

> Esta seccion es **documentacion para implementacion futura** (no se incluye codigo aun). Sirve como referencia para cuando se requiera instrumentacion real en produccion.

### 11.1 Que es OpenTelemetry y por que importa aca

OpenTelemetry (OTel) es el estandar de la industria para **observabilidad** (telemetry data: traces, metrics, logs). En MLOps, complementa el drift detection con visibilidad operacional: cuanto tarda una prediccion, cuantos errores 5xx hubo, que distribucion de generos estamos prediciendo, etc.

**Beneficio clave**: correlacionar drift detection (estadistico) con telemetria operacional (latencia, errores) para entender **por que** un modelo degrada, no solo **que** degrada.

### 11.2 Que instrumentar en este proyecto

| Componente | Tipo | Que capturar |
|---|---|---|
| FastAPI `app/main.py` | **Tracing** | Spans automaticos por request (latencia, status code, ruta) via `opentelemetry-instrumentation-fastapi` |
| `predict_genre()` | **Tracing (custom span)** | Tiempo de `model.predict`, tiempo de carga del modelo, feature vector usado |
| `POST /predict` | **Metrics** | Counter `predictions_total{genre=...}`, Histogram `prediction_latency_seconds`, Counter `errors_total{type=...}` |
| Middleware `log_requests` | **Logging** | Logs estructurados correlacionados con `trace_id` y `span_id` |
| MLflow runs | **Correlation** | Tag del run_id de MLflow en el span para trazabilidad modelo<->prediccion |

### 11.3 Stack tecnologico recomendado

| Componente | Libreria | Proposito |
|---|---|---|
| API | `opentelemetry-instrumentation-fastapi` | Auto-instrumentacion de FastAPI |
| API | `opentelemetry-instrumentation-httpx` | Si la API llama a otros servicios |
| API | `opentelemetry-instrumentation-logging` | Correlacionar logs con traces |
| Metrics | `opentelemetry-exporter-prometheus` o `opentelemetry-exporter-otlp` + Prometheus | Metricas para alerting/Grafana |
| Tracing | `opentelemetry-exporter-otlp` + Jaeger o Tempo | Visualizacion de traces distribuidos |
| Collector (opcional) | `opentelemetry-collector-contrib` | Aggregar y exportar a multiples backends |

### 11.4 Metricas y logs a correlacionar con drift

El middleware actual escribe `logs/api_requests.jsonl` con los features de cada request. Con OTel se anade:

1. **`prediction_distribution{genre}`** (Gauge) — distribucion acumulada de generos predichos. Un cambio brusco aqui es un **proxy de concept drift** (mas rapido de detectar que KS sobre features).
2. **`input_feature_stats{feature,stat}`** (Gauge) — rolling mean/std de features de entrada. Alimenta el drift detection en tiempo real sin esperar a un job batch.
3. **`model_confidence_histogram`** (Histogram) — si la confidence media baja, el modelo esta dudando (segn de drift).
4. **`prediction_latency_p95`** (Gauge derivado) — spike de latencia puede indicar data de input inusualmente compleja.

**Patron recomendado**: emitir estas metricas en `predict_genre()` con el SDK de OTel, exportarlas a Prometheus, y crear un dashboard Grafana con alertas (e.g., `prediction_distribution shift > 30%` en 1h).

### 11.5 Integracion con el middleware existente

El middleware `log_requests` puede enriquecerse para:
- Extraer el `trace_context` del header entrante (`traceparent`) y propagarlo.
- Anadir `trace_id` y `span_id` a cada linea JSONL de `api_requests.jsonl`.
- Asi, el script `analyze_drift.py --mode online` puede correlacionar drifts detectados con traces especificos de OTel para debugging.

### 11.6 Como se integraria con pipelines CI/CD

- `concept-drift-check.yml` puede exportar metricas de drift a Prometheus (no solo al JSON) y crear un dashboard por corrida.
- `docker-build.yml` puede incluir el OTel Collector como sidecar o servicio en `docker-compose`.
- Un pipeline futuro `otel-validate.yml` podria verificar que la API instrumentada emite las metricas esperadas (contract testing).

### 11.7 Estimacion de esfuerzo

- Setup basico (FastAPI auto-instrumentation + OTLP exporter): 2-4 horas.
- Metricas custom (prediction distribution, input stats): 4-6 horas.
- Dashboards Grafana + alertas: 4-8 horas.
- Tracing custom spans en `predict_genre()`: 2-3 horas.
- **Total para implementacion production-ready**: 2-3 dias.

### 11.8 Referencias y documentacion

- [OpenTelemetry Python docs](https://opentelemetry.io/docs/languages/python/)
- [FastAPI instrumentation](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html)
- [OTel Collector](https://opentelemetry.io/docs/collector/)
- [Prometheus + OTel](https://prometheus.io/docs/guides/opentelemetry/)

---

## Anexo A — Snippets YAML de Pipelines CI/CD

> Los siguientes archivos son **pegables tal cual** en `.github/workflows/`.

### Anexo A.1 — `docker-build.yml`

```yaml
name: Docker Build & Publish

on:
  push:
    branches: [main]
    tags: ['v*.*.*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,format=short

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ./model_serving
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            MLFLOW_TRACKING_URI=${{ secrets.MLFLOW_TRACKING_URI }}
```

**Secrets requeridos**: `GITHUB_TOKEN` (automatico), `MLFLOW_TRACKING_URI` (opcional, default `http://localhost:5000`).

### Anexo A.2 — `concept-drift-check.yml`

```yaml
name: Concept Drift Check

on:
  schedule:
    - cron: '0 6 * * 1'  # Cada lunes 06:00 UTC
  workflow_dispatch:
  pull_request:
    paths:
      - 'data_pipeline/**'
      - 'model_serving/**'
      - 'drift_monitoring/**'
      - '.github/workflows/concept-drift-check.yml'

jobs:
  drift-check:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r data_pipeline/requirements.txt
          pip install -r drift_monitoring/requirements.txt

      - name: Run batch drift analysis
        run: |
          python drift_monitoring/src/analyze_drift.py \
            --mode batch \
            --train_data data_pipeline/data/train.csv \
            --prod_data data_pipeline/data/prod_sim.csv \
            --output drift_report.json

      - name: Upload drift report
        uses: actions/upload-artifact@v4
        with:
          name: drift-report
          path: drift_report.json
          retention-days: 30

      - name: Check drift threshold
        run: |
          DRIFT_PCT=$(python -c "import json; r=json.load(open('drift_report.json')); print(r.get('drift_percentage', 0))")
          echo "Drift percentage: ${DRIFT_PCT}%"
          if [ "$(python -c "print(int(float('$DRIFT_PCT') > 50))")" = "1" ]; then
            echo "::warning::Drift exceeds 50% threshold. Consider retraining."
          fi

      - name: Comment on PR
        if: github.event_name == 'pull_request' && always()
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          header: drift-report
          path: drift_report.json
```

**Valor**: Falla el build si drift > 50%, alerta con `::warning::`, sube el reporte como artefacto descargable.

### Anexo A.3 — `dvc-validate.yml`

```yaml
name: DVC Data & Pipeline Validation

on:
  pull_request:
    paths:
      - 'data_pipeline/dvc.yaml'
      - 'data_pipeline/params.yaml'
      - 'data_pipeline/songs.csv.dvc'
      - 'data_pipeline/src/**'

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install DVC
        run: pip install dvc>=3.0.0

      - name: Validate DVC DAG
        working-directory: data_pipeline
        run: |
          dvc dag --dot | dot -Tsvg > dvc_dag.svg
          echo "DAG validated successfully"

      - name: Validate params.yaml schema
        working-directory: data_pipeline
        run: |
          python -c "
          import yaml
          with open('params.yaml') as f:
              params = yaml.safe_load(f)
          assert 'train' in params, 'Missing train section'
          assert 'logistic_regression' in params['train'], 'Missing LR params'
          assert 'xgboost' in params['train'], 'Missing XGBoost params'
          print('params.yaml schema OK')
          "

      - name: Check DVC file consistency
        working-directory: data_pipeline
        run: |
          dvc status || echo "::notice::DVC reports changes (expected if data not yet present)"

      - name: Upload DAG visualization
        uses: actions/upload-artifact@v4
        with:
          name: dvc-dag
          path: data_pipeline/dvc_dag.svg
```

**Valor**: Garantiza que el DAG de DVC esta bien formado, que `params.yaml` tiene la estructura esperada, y visualiza la pipeline.

---

## Anexo B — Variables de Entorno y Secretos

### Variables locales (`.env`)

```bash
# .env (basado en .env.example)
MLFLOW_TRACKING_URI=http://localhost:5000

# Opcionales para produccion
DVC_REMOTE_URL=s3://your-bucket/dvc-storage
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
```

### Secrets de GitHub requeridos por pipelines

| Secret | Usado por | Descripcion |
|---|---|---|
| `GITHUB_TOKEN` | `docker-build.yml` | Automatico, no requiere config |
| `MLFLOW_TRACKING_URI` | `docker-build.yml` | URI del servidor MLflow reachable desde el runner (e.g., IP publica o `http://host.docker.internal:5000`) |
| `DVC_REMOTE_URL` | `dvc-validate.yml` (plus) | Si se quiere hacer `dvc pull` en CI |

### Donde configurar

`Settings -> Secrets and variables -> Actions -> New repository secret`

---

## Anexo C — Glosario MLOps

| Termino | Definicion |
|---|---|
| **DVC** | Data Version Control — Git para data y modelos, permite versionar datasets grandes sin guardar binarios en Git |
| **MLflow** | Plataforma open-source para gestionar el ciclo de vida ML: tracking de experimentos, Model Registry, deployment |
| **Model Registry** | Componente de MLflow para versionar modelos registrados; permite aliases y stages |
| **`@champion` alias** | Alias movil que apunta a la "version actual en produccion" del modelo; se reasigna sin cambiar el nombre |
| **Data drift** | Cambio en la distribucion de las features de entrada entre train y produccion |
| **Concept drift** | Cambio en la relacion feature->target; el modelo "desaprende" el patron |
| **KS test (Kolmogorov-Smirnov)** | Test no parametrico que compara dos distribuciones; devuelve estadistico y p-value |
| **`p_value < 0.05`** | Umbral de significancia comun: si p < 0.05, las distribuciones son significativamente diferentes |
| **Champion/Challenger** | Patron donde el modelo en produccion (champion) compite con candidatos (challengers) antes de ser promovidos |
| **Self-contained image** | Imagen Docker que incluye todas sus dependencias (incluido el modelo); no requiere servicios externos en runtime |
| **Build-time vs Runtime** | Build-time = durante `docker build`; Runtime = cuando el contenedor ya esta corriendo |
| **OWLP / OTLP** | OpenTelemetry Protocol — formato estandar para exportar telemetry data |
| **Span** | Unidad de trabajo en distributed tracing; tiene nombre, duracion, atributos |
| **Trace** | Conjunto de spans relacionados que representan un request end-to-end |

---

## Anexo D — Comandos Utiles Consolidados

### DVC

```bash
cd data_pipeline
dvc repro                    # Ejecuta el pipeline completo
dvc dag                      # Visualiza la DAG en texto
dvc dag --dot | dot -Tsvg > dag.svg   # DAG en SVG
dvc status                   # Que stages necesitan re-ejecucion
dvc status songs.csv.dvc     # Verifica hash del dataset
dvc pull                    # Descarga data desde remote
dvc push                    # Sube data a remote
dvc metrics show            # Muestra metricas tracked
dvc params diff            # Diff de hiperparametros entre runs
```

### MLflow

```bash
# Iniciar servidor (en otra terminal)
mlflow server --host 0.0.0.0 --port 5000

# Listar modelos registrados
mlflow models list

# Descargar modelo (usado en Dockerfile)
mlflow models download -m "models:/spotify-genre-classifier@champion" -d ./models --no-directory

# UI web
# Abrir http://localhost:5000 en el navegador
```

### API

```bash
cd model_serving
uvicorn app.main:app --reload --port 8000   # Dev con autoreload
uvicorn app.main:app --host 0.0.0.0 --port 8000  # Produccion local

# Probar endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"danceability":0.7,"energy":0.8,"key":5,"loudness":-5.0,"mode":1,"speechiness":0.05,"acousticness":0.1,"instrumentalness":0.0,"liveness":0.2,"valence":0.6,"tempo":120.0,"duration_ms":240000}'
```

### Docker

```bash
# Build
docker build -t spotify-api:latest ./model_serving
# Build con override de MLflow URI
docker build --build-arg MLFLOW_TRACKING_URI=http://host.docker.internal:5000 -t spotify-api:latest ./model_serving

# Run
docker run -p 8000:8000 spotify-api:latest
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 spotify-api:latest

# Inspeccionar
docker images | grep spotify-api
docker logs <container_id>
```

### Testing

```bash
# Lint
flake8 .

# Tests
pytest data_pipeline/tests -v
pytest model_serving/tests -v
pytest drift_monitoring/ -v  # Si se agregan tests

# Con coverage
pytest --cov=data_pipeline/src --cov=model_serving/app
```

### Drift Monitoring

```bash
cd drift_monitoring
python src/analyze_drift.py \
  --mode batch \
  --train_data ../data_pipeline/data/train.csv \
  --prod_data ../data_pipeline/data/prod_sim.csv \
  --output batch_drift_report.json

python src/analyze_drift.py \
  --mode online \
  --train_data ../data_pipeline/data/train.csv \
  --api_logs ../model_serving/logs/api_requests.jsonl \
  --output online_drift_report.json

# Ver reporte
cat batch_drift_report.json | python -m json.tool | head -50
```

### Git / GitHub

```bash
# Workflow basico
git checkout -b solution/<tu-nombre>
git add .
git commit -m "feat: implement data pipeline process step"
git push origin solution/<tu-nombre>

# Abrir PR desde la UI de GitHub
# Titulo: [Homework] <Tu Nombre Completo>
# Base: main del repo del curso
```

### Kaggle

```bash
# Setup (una vez)
mkdir -p ~/.kaggle  # Linux/Mac
# En Windows: mkdir $env:USERPROFILE\.kaggle
# Copiar kaggle.json ahi
chmod 600 ~/.kaggle/kaggle.json  # Linux/Mac

# Descargar dataset
kaggle datasets download -d serkantysz/550k-spotify-songs-audio-lyrics-and-genres
unzip 550k-spotify-songs-audio-lyrics-and-genres.zip
mv songs.csv data_pipeline/
```

---

## Anexo E — Mapeo Punto-a-Punto con la Rubrica (20 pts)

> Este anexo es la **tabla de auditoria final**: cada item de `GRADING_RUBRIC.md` con su comando o archivo de verificacion, y la accion concreta que asegura el punto.

### §1. Data Pipeline — 6 pts

| ID Rubrica | Pts | Criterio exacto | Como verificarlo | Que hacer concretamente |
|---|---|---|---|---|
| **§1.1** | 0.5 | `songs.csv` MD5 matches `0e71e2c46244acac485bd8c245aa6e56` | `cd data_pipeline && dvc status songs.csv.dvc` → limpio / "Data and pipelines are up to date" | Descargar CSV correcto de Kaggle, comparar MD5 |
| **§1.1** | 0.5 | `dvc repro` produce `data/raw.csv` con column count correcto | `head -1 data/raw.csv \| tr ',' '\n' \| wc -l` → debe ser ~17-19 cols (Kaggle dataset tiene 17-19) | Correr `dvc repro`; verificar shape con `python -c "import pandas as pd; print(pd.read_csv('data/raw.csv').shape)"` |
| **§1.2** | 0.5 | Temporal split: `year <= 2010` → train, `year > 2010` → prod_sim (boundary exacto) | `pytest data_pipeline/tests/test_process.py::test_process_data_year_boundary_condition` | Implementar split con `<=` (no `<`) en `process.py:39-48` |
| **§1.2** | 0.5 | Ambos `data/train.csv` y `data/prod_sim.csv` producidos | `ls -la data_pipeline/data/{train,prod_sim}.csv` | Implementar `to_csv(..., index=False)` con `os.makedirs(..., exist_ok=True)` |
| **§1.2** | 0.5 | Audio features + `genre` presentes en ambos outputs | `pytest data_pipeline/tests/test_process.py::test_process_data_preserves_audio_features` | No dropear columnas en el split; usar solo `df[df.year <= thr]` y `df[df.year > thr]` |
| **§1.3** | 0.5 | Carga training data y target `genre` correctamente | Abrir MLflow UI tras train; los runs deben mostrar `model.fit` exitoso | `pd.read_csv(data_path)`, `y = df["genre"]`, `X = df.drop(["genre", "year"], axis=1, errors='ignore')` |
| **§1.3** | 0.5 | Entrena 2+ modelos (LR + otro) | `params['train']` tiene `logistic_regression` y `xgboost` (ya esta en `params.yaml`) | Loop `for model_name, model_params in train_params.items():` instanciando `LogisticRegression` o `xgb.XGBClassifier` |
| **§1.3** | 0.5 | Loguea params y accuracy a MLflow por cada run | En la UI: cada run tiene `Params` (C, max_iter, etc.) y `Metrics` (accuracy) | `with mlflow.start_run(run_name=model_name):` + `mlflow.log_params(model_params)` + `mlflow.log_metric("accuracy", acc)` |
| **§1.3** | 0.5 | Runs visibles en MLflow UI con naming y artefactos | `http://localhost:5000` → 2 runs con nombres `logistic_regression` y `xgboost`; cada uno con `model/` artifact | Usar `run_name=model_name`; `mlflow.sklearn.log_model(model, "model")` o `mlflow.xgboost.log_model(model, "model")` |
| **§1.4** | 0.5 | Encuentra el mejor modelo por accuracy | `evaluate.py` ya tiene `order_by=["metrics.accuracy DESC"]`; verificar que `metrics.json` tiene `best_accuracy` razonable (e.g., > 0.3) | Sin cambios (el esqueleto ya lo hace); solo completar el registro |
| **§1.4** | 0.5 | Registra en Model Registry con alias `champion` | `mlflow models list` debe mostrar `spotify-genre-classifier`; en UI: `Aliases → champion → v1` | Agregar 2 lineas: `client.create_model_version(...)` + `client.set_registered_model_alias(name, "champion", v)` |
| **§1.5** | 0.5 | `dvc repro` corre sin errores y produce outputs esperados | `cd data_pipeline && dvc repro` → exit 0, `data/{raw,train,prod_sim}.csv` y `metrics.json` existen | `dvc.yaml` ya esta completo; solo ejecutar |

### §2. Model Serving — 5 pts

| ID Rubrica | Pts | Criterio exacto | Como verificarlo | Que hacer concretamente |
|---|---|---|---|---|
| **§2.1** | 1 | `GET /health` retorna respuesta correcta | `curl http://localhost:8000/health` → `{"status":"healthy"}` 200 | `@app.get("/health") def health(): return {"status":"healthy"}` |
| **§2.1** | 1 | `POST /predict` acepta `SpotifyFeatures` valido y retorna prediccion | `pytest model_serving/tests/test_api.py::test_predict_endpoint_valid_payload` | El esqueleto ya tiene el endpoint; falta implementar `predict_genre()` real (no placeholder) |
| **§2.1** | 1 | Request logging escribe a `logs/api_requests.jsonl` | Hacer 1 `POST /predict`, luego `ls model_serving/logs/` y `cat model_serving/logs/api_requests.jsonl` | Implementar middleware con `await request.body()`, append JSON line, reconstruir `Request` |
| **§2.2** | 1 | `SpotifyFeatures` con audio features y tipos correctos | `pytest model_serving/tests/test_api.py` → payload con 12 campos pasa validacion | Definir los 12 campos en Pydantic con tipos correctos (ver §5.4) |
| **§2.3** | 0.5 | Dockerfile build sin errores | `docker build -t test ./model_serving` → SUCCESS | (a)+(b) implementados correctamente; el build no debe fallar por dependencias faltantes |
| **§2.3** | 0.5 | Incluye step para descargar `@champion` de MLflow | `docker run test cat /app/models/MLmodel` → existe (si MLflow estaba corriendo durante build) | `ARG MLFLOW_TRACKING_URI=...` + `RUN mlflow models download -m models:/spotify-genre-classifier@champion -d ./models --no-directory` |

### §3. Drift Monitoring — 3 pts

| ID Rubrica | Pts | Criterio exacto | Como verificarlo | Que hacer concretamente |
|---|---|---|---|---|
| **§3.1** | 0.5 | `--mode batch` carga `data/train.csv` y `data/prod_sim.csv` correctamente | Correr batch y verificar que logs dicen "[BATCH] Loading training data from..." y "...production data from..." | `analyze_batch_drift` ya esta en el esqueleto; solo implementar `run_ks_analysis` |
| **§3.1** | 0.5 | KS test por cada audio feature con `scipy.stats.ks_2samp` | `cat batch_drift_report.json \| python -m json.tool` → 12 entries en `details` (uno por audio feature) | Loop por `features_to_test` con `ks_2samp(train_values, prod_values)` |
| **§3.1** | 0.5 | `drift_report.json` con `ks_statistic`, `p_value`, `drift_detected`, `status` | Validar JSON: `python -c "import json; r=json.load(open('batch_drift_report.json')); assert 'status' in r; assert all('ks_statistic' in v for v in r['details'].values())"` | Poblar `drift_results["details"][feature] = {ks_statistic, p_value, drift_detected, train_mean, prod_mean}` |
| **§3.2** | 0.5 | `--mode online` carga `data/train.csv` y `logs/api_requests.jsonl` | Correr online tras 1+ POST; logs dicen "[ONLINE] Loading API logs from..." | `analyze_online_drift` ya parsea JSONL; solo implementar `run_ks_analysis` |
| **§3.2** | 0.5 | Parsea JSONL linea por linea y construye DataFrame | `head model_serving/logs/api_requests.jsonl \| python -c "import sys,json; [print(json.loads(l).keys()) for l in sys.stdin]"` debe mostrar audio features | El esqueleto ya hace `for line in f: api_logs.append(json.loads(line))` |
| **§3.2** | 0.5 | Reusa `run_ks_analysis` para ambos modos | `analyze_online_drift` debe terminar con `return run_ks_analysis(train_df, api_df, output_path)` | Sin cambios estructurales; solo completar la funcion compartida |

### §4. Testing & CI/CD — 4 pts

| ID Rubrica | Pts | Criterio exacto | Como verificarlo | Que hacer concretamente |
|---|---|---|---|---|
| **§4.1** | 1 | `pytest data_pipeline/tests` pasa (todas las aserciones) | `cd data_pipeline && pytest -v` → 6 passed | `test_load.py` (3) + `test_process.py` (3); tests ya pasan con codigo actual siempre que los TODOs esten implementados |
| **§4.1** | 1 | `pytest model_serving/tests` pasa (todas las aserciones) | `cd model_serving && pytest -v` → 3 passed | `test_api.py` (3); requiere `SpotifyFeatures` con los 12 campos y `/health` implementados |
| **§4.2** | 1 | `flake8 .` sin violaciones mayores (warnings OK) | `flake8 .` → exit 0 (warnings pueden aparecer, no errores) | Respetar `max-line-length=100`; no imports no usados; `per-file-ignores` ya exime a `main.py:F401` |
| **§4.3** | 1 | CI pipeline pasa en PR (linter + todos los tests) | Abrir PR → check verde en la pestaña "Actions" | El `ci.yml` ya corre flake8 + pytests; asegurarse de que pasen localmente antes del push |

### §5. Documentation & Code Quality — 2 pts

| ID Rubrica | Pts | Criterio exacto | Como verificarlo | Que hacer concretamente |
|---|---|---|---|---|
| **§5.1** | 1 | Todos los `TODO comments` direccionados; estilo Python correcto | `grep -r "TODO\|FIXME" data_pipeline/src model_serving/app drift_monitoring/src` → vacio (excepto justificados) | Reemplazar cada `TODO` con implementacion real, o comentar justificando por que se deja |
| **§5.2** | 0.5 | README claro e instrucciones followable | Lectura humana del README; tiene secciones para download, setup, run, test, submit | README ya esta completo; verificar que no este desactualizado respecto a cambios |
| **§5.2** | 0.5 | Setup end-to-end funciona (download → process → train → evaluate) | Seguir la `Orden de Ejecucion Sugerido` (§9) en una maquina limpia | Verificar que cada comando corre sin error en secuencia |

### Resumen de puntos y prioridad

| Rubrica | Total | Como | Cuando se asegura |
|---|---|---|---|
| §1 Data Pipeline | 6 | Implementacion + ejecucion | Tras `dvc repro` exitoso y verificar MLflow UI |
| §2 Model Serving | 5 | Implementacion + `docker build/run` | Tras `pytest model_serving` en verde y curl `/predict` OK |
| §3 Drift Monitoring | 3 | Implementacion + 2 corridas | Tras `batch_drift_report.json` y `online_drift_report.json` generados |
| §4 Testing & CI/CD | 4 | `pytest` + `flake8` + PR verde | Tras abrir PR con CI en verde |
| §5 Docs & Quality | 2 | `grep TODO` vacio + setup e2e | Tras limpiar TODOs y verificar setup |
| **TOTAL** | **20** | | |

### Checklist de auto-evaluacion pre-PR

```bash
# 1. Lint limpio
flake8 . && echo "[OK] flake8"

# 2. Tests pasando
pytest data_pipeline/tests -v
pytest model_serving/tests -v

# 3. Sin TODOs residuales
grep -rn "TODO\|FIXME\|XXX" data_pipeline/src model_serving/app drift_monitoring/src 2>/dev/null && echo "[WARN] TODOs found" || echo "[OK] no TODOs"

# 4. DVC pipeline funcional
cd data_pipeline && dvc status && echo "[OK] DVC"

# 5. Drift reports generados
test -f batch_drift_report.json && echo "[OK] batch drift" || echo "[WARN] batch drift missing"
test -f online_drift_report.json && echo "[OK] online drift" || echo "[WARN] online drift missing"

# 6. Modelos registrados
curl -s http://localhost:5000/api/2.0/mlflow/registered-models/get-latest-versions/spotify-genre-classifier 2>/dev/null | python -c "import json,sys; d=json.load(sys.stdin); print('[OK] champion' if any(v.get('aliases', []) == ['champion'] for v in d.get('model_versions', [])) else '[WARN] no champion')"

# 7. Docker build (opcional pero recomendado)
docker build -t spotify-api-test ./model_serving && echo "[OK] docker build"
```

Si todos los items salen en `[OK]`, el PR deberia obtener 20/20.

---

## Anexo F — Submission Checklist para el PR

> Copia este bloque (con checks) en la **descripcion del PR** para que el evaluador vea rapidamente tu cobertura.

```markdown
## Submission Checklist

### Data Pipeline (6 pts)
- [ ] `songs.csv` MD5 matches `0e71e2c46244acac485bd8c245aa6e56` (§1.1)
- [ ] `dvc repro` produces `data/raw.csv` with correct columns (§1.1)
- [ ] Temporal split: `year <= 2010` → train, `year > 2010` → prod_sim (§1.2)
- [ ] Both `data/train.csv` and `data/prod_sim.csv` are produced (§1.2)
- [ ] Audio features + `genre` present in both outputs (§1.2)
- [ ] Loads training data and target `genre` correctly (§1.3)
- [ ] Trains 2+ different models (Logistic Regression + XGBoost) (§1.3)
- [ ] Logs parameters and accuracy to MLflow for each run (§1.3)
- [ ] Runs visible in MLflow UI with naming and artifacts (§1.3)
- [ ] Best model found by accuracy metric (§1.4)
- [ ] Best model registered with `champion` alias (§1.4)
- [ ] `dvc repro` runs end-to-end without errors (§1.5)

### Model Serving (5 pts)
- [ ] `GET /health` returns `{"status":"healthy"}` (§2.1)
- [ ] `POST /predict` accepts `SpotifyFeatures` and returns prediction (§2.1)
- [ ] Request logging writes to `logs/api_requests.jsonl` (§2.1)
- [ ] `SpotifyFeatures` includes all 12 audio features with correct types (§2.2)
- [ ] Dockerfile builds without errors (§2.3)
- [ ] Dockerfile includes step to download `@champion` from MLflow (§2.3)

### Drift Monitoring (3 pts)
- [ ] Batch mode loads `data/train.csv` and `data/prod_sim.csv` correctly (§3.1)
- [ ] Kolmogorov-Smirnov test runs for each audio feature (§3.1)
- [ ] `drift_report.json` contains per-feature KS results and overall status (§3.1)
- [ ] Online mode loads `data/train.csv` and `logs/api_requests.jsonl` correctly (§3.2)
- [ ] Online mode parses JSONL and builds DataFrame of production features (§3.2)
- [ ] Online mode reuses `run_ks_analysis` from batch mode (§3.2)

### Testing & CI/CD (4 pts)
- [ ] `pytest data_pipeline/tests` passes (§4.1)
- [ ] `pytest model_serving/tests` passes (§4.1)
- [ ] `flake8 .` shows no major style violations (§4.2)
- [ ] GitHub Actions CI passes on this PR (green checkmark) (§4.3)

### Documentation & Code Quality (2 pts)
- [ ] All TODO comments in code are addressed or justified (§5.1)
- [ ] README is clear and instructions are followable (§5.2)
- [ ] Setup works end-to-end (download → process → train → evaluate) (§5.2)

### Bonus/Advanced (optional, +2 pts max)
- [ ] `docker-build.yml` pipeline added
- [ ] `concept-drift-check.yml` pipeline added
- [ ] `dvc-validate.yml` pipeline added
- [ ] OpenTelemetry documentation (sec 11) referenced for future work
```

---

## 13. Portfolio en AWS: Analisis de Costos y Arquitectura

> **Esta seccion esta pensada para que tengas un portafolio production-ready en AWS.** Dado que conoces AWS, te doy opciones concretas con precios reales (region us-east-1, enero 2026) y arquitecturas de referencia.

### 13.1 El "game changer" que descubri investigando: $200 en credits

AWS ofrece actualmente **$100 USD en credits inmediatamente + hasta $100 USD adicionales** al crear una cuenta nueva, validos por **6 meses** en la mayoria de servicios. Esto significa que **puedes correr TODO este lab en AWS gratis** durante medio ano si eres cuenta nueva.

**Para que te alcanza con los $200**:

| Concepto | Costo mensual aprox. | Meses cubierto con $200 |
|---|---|---|
| Arquitectura minima (1 EC2 t3.micro + S3 + ECR) | ~$8-12 | 16-25 meses |
| Arquitectura media (EC2 MLflow + ECS Fargate API + S3) | ~$40-60 | 3-5 meses |
| Arquitectura completa (SageMaker Studio + Endpoints + todo) | ~$150-300 | <1 mes (usar para demos cortas) |

**Conclusion**: con los $200, **corre la arquitectura media (recomendada) durante 4-5 meses** sin pagar un centavo. Es el sweet spot para portfolio.

### 13.2 Tres opciones de arquitectura (de menor a mayor inversion)

#### Opcion A — "Light Portfolio" (~ $10-15/mes post-free-tier)

**Filosofia**: Replicar exactamente lo que tenemos local, pero en AWS.

```
                    +-------------------+
                    |   GitHub Actions  |
                    |  (lint + tests)   |
                    +---------+---------+
                              |
                              v
+-------------+        +------+-------+        +-------------+
| Kaggle API  |------->|  EC2 t3.micro|<------>| Docker Hub  |
|  (download) |        |  (MLflow +   |        |  (opcional) |
+-------------+        |   Docker)    |        +------+------+
                        +------+-------+               |
                               |                       |
                               v                       v
                        +------+-------+        +------+------+
                        |  S3 bucket   |        | FastAPI en |
                        | (DVC remote) |        |  EC2 o ECS |
                        +--------------+        +-------------+
```

**Stack concreto**:

| Servicio | SKU | Costo/mes (us-east-1) |
|---|---|---|
| EC2 t3.micro (1 vCPU, 1 GB RAM) | on-demand | $7.59 (~$0.0104/hora) |
| EBS gp3 20 GB (volumen EC2) | storage | $1.60 |
| S3 Standard (10 GB dataset + modelos) | storage | $0.23 |
| Data transfer out (5 GB/mes) | traffic | $0.45 |
| Elastic IP (asociada) | static IP | $0.00 (si esta asociada) |
| **TOTAL** | | **~$9.87/mes** |

**Setup** (resumido):
```bash
# 1. Lanzar EC2 t3.micro con Amazon Linux 2, 20 GB EBS, security group con puertos 22 y 5000 abiertos a tu IP
aws ec2 run-instances --image-id ami-0c02fb55956d7a4f6 --instance-type t3.micro \
  --key-name mlops-key --security-groups mlops-sg --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20}}]'

# 2. SSH a la instancia e instalar MLflow + dependencias
ssh ec2-user@<public-ip>
sudo yum install python3 git -y
pip3 install mlflow boto3 dvc[s3]

# 3. Crear bucket S3 para MLflow artifacts
aws s3 mb s3://mlops-spotify-portfolio-mlflow

# 4. Correr MLflow server (accesible en http://<public-ip>:5000)
mlflow server --host 0.0.0.0 --port 5000 \
  --default-artifact-root s3://mlops-spotify-portfolio-mlflow/artifacts \
  --backend-store-uri sqlite:///mlflow.db &

# 5. Instalar DVC y configurar S3 remote (en la instancia o tu maquina)
dvc remote add -d myremote s3://mlops-spotify-portfolio-dvc
dvc push
```

**Pros**:
- Barato, simple, replica el entorno local.
- Suficiente para demos de portfolio.
- Libre control del ambiente (puedes instalar lo que quieras).

**Contras**:
- Single point of failure (si EC2 se cae, MLflow no responde).
- MLflow y API comparten la misma EC2 (no ideal para concurrencia).
- Seguridad: puertos expuestos directamente a internet.

---

#### Opcion B — "Production-Style Portfolio" (~ $40-60/mes post-free-tier) — **RECOMENDADA**

**Filosofia**: Separar responsabilidades como en un sistema real: MLflow aislado, API serverless, storage dedicado, monitoreo con CloudWatch.

```
+----------------+        +-----------------+        +------------------+
|  GitHub Repo   |------->| GitHub Actions  |------->|  Amazon ECR      |
|  (PR trigger)  |        | (lint/test/     |        |  (docker images) |
+----------------+        |  build/push)    |        +---------+--------+
                           +-----------------+                  |
                                                                 v
+----------------+        +-----------------+        +------------------+
|  CloudWatch    |<-------|  ECS Fargate    |<-------|  ALB (opcional)  |
|  Logs/Metrics  |        |  (FastAPI)      |        |  $16/mes extra   |
|  ($0.50/mes)   |        |  0.25 vCPU      |        +------------------+
+----------------+        |  0.5 GB RAM     |
                          +--------+--------+
                                   |
                                   v  descarga @champion en build
                          +--------+--------+        +------------------+
                          |  EC2 t3.micro   |        |  S3 buckets      |
                          |  (MLflow server)|<------>|  mlops-dvc-data  |
                          |  $7.59/mes      |        |  mlops-mlflow-   |
                          +-----------------+        |  artifacts       |
                                                     +------------------+
```

**Stack concreto**:

| Servicio | SKU | Costo/mes (us-east-1) |
|---|---|---|
| EC2 t3.micro (MLflow server) | on-demand 24/7 | $7.59 |
| EBS gp3 20 GB (EC2) | storage | $1.60 |
| ECS Fargate (FastAPI, 0.25 vCPU, 0.5 GB) | 24/7 | $4.69 (vCPU) + $1.84 (mem) = $6.53 |
| ECR (1 imagen ~500 MB) | storage | $0.05 |
| S3 DVC remote (10 GB) | storage | $0.23 |
| S3 MLflow artifacts (5 GB) | storage | $0.12 |
| CloudWatch Logs (5 GB ingested) | logs | $2.50 |
| CloudWatch Metrics (custom + AWS) | metrics | $0.50 (free tier cubre lo basico) |
| Data transfer (5 GB/mes out) | traffic | $0.45 |
| ECR data transfer (sin ALB) | $0 | $0.00 |
| Secrets Manager (1 secret: DB URL) | $0.40/secret/mes | $0.40 |
| **TOTAL** | | **~$19.96/mes** |

Espera, recalculo: **~$20/mes** (no $40-60). Lo que sube el costo seria agregar **ALB** (Application Load Balancer, ~$16/mes) y **SageMaker Model Monitor** para drift detection en cloud.

Con ALB opcional: **~$36/mes**.

**Setup** (resumido):
```bash
# 1. EC2 para MLflow (mismo que Opcion A)

# 2. Crear ECR repo y pushear imagen
aws ecr create-repository --repository-name spotify-api
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
docker tag spotify-api:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/spotify-api:latest
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/spotify-api:latest

# 3. Crear cluster ECS Fargate + task definition
aws ecs create-cluster --cluster-name mlops-cluster

# Task definition JSON (simplificado):
cat > task-def.json <<EOF
{
  "family": "spotify-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [{
    "name": "api",
    "image": "<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/spotify-api:latest",
    "portMappings": [{"containerPort": 8000}],
    "environment": [
      {"name": "MLFLOW_TRACKING_URI", "value": "http://<EC2-IP>:5000"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/spotify-api",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
EOF
aws ecs register-task-definition --cli-input-json file://task-def.json

# 4. Crear service con desired count 1
aws ecs create-service --cluster mlops-cluster --service-name spotify-api \
  --task-definition spotify-api --desired-count 1 --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

**Pros**:
- Separacion clara de responsabilidades (mejor para mostrar en portafolio).
- API serverless = auto-scaling "gratis" (pagas solo por uso).
- Logs centralizados en CloudWatch.
- Si la API crashea, ECS la reinicia automaticamente.
- Se ve **profesional** en un portafolio (grafico de arquitectura real).

**Contras**:
- Mas servicios que configurar (10 vs 3 de Opcion A).
- Networking mas complejo (VPC, subnets, security groups).
- Debugging mas dificil (logs distribuidos).

---

#### Opcion C — "SageMaker Full Stack" (~ $100-300/mes)

**Filosofia**: Usar todos los servicios administrados de SageMaker.

```
+----------------+        +-----------------+        +------------------+
|  SageMaker     |        |  SageMaker      |        |  SageMaker       |
|  Studio        |<------>|  MLflow         |<-------|  Endpoints       |
|  (notebooks +  |        |  Tracking       |        |  (Real-time)     |
|   experiments) |        |  Server         |        |  ml.t2.medium    |
+----------------+        +-----------------+        +------------------+
        |                          |                          |
        v                          v                          v
+----------------+        +-----------------+        +------------------+
| SageMaker      |        |  SageMaker      |        |  SageMaker       |
| Pipelines      |        |  Feature Store  |        |  Model Monitor   |
| (orquestacion) |        |  (opcional)     |        |  (drift detect)  |
+----------------+        +-----------------+        +------------------+
```

**Costos estimados** (usando las tarifas reales de AWS):

| Componente | Pricing | Costo/mes (uso bajo) |
|---|---|---|
| SageMaker Studio Notebook (ml.t3.medium, 4h/dia) | $0.05/h | $6.00 (120h) |
| SageMaker MLflow Tracking (Small, 24/7) | $0.60/h | $432 (si 24/7) — **carisimo** |
| SageMaker MLflow Tracking (Small, 4h/dia) | $0.60/h | $72 |
| SageMaker Training (ml.m5.xlarge, 30 min/dia) | $0.23/h | $3.45 (15h) |
| SageMaker Real-time Endpoint (ml.t2.medium, 24/7) | $0.065/h | $46.80 |
| SageMaker Model Monitor (ml.m5.xlarge, 1h/dia) | $0.23/h | $6.90 (30h) |
| S3 (10 GB) | $0.023/GB | $0.23 |
| CloudWatch Logs (5 GB) | $0.50/GB | $2.50 |
| **TOTAL bajo uso** | | **~$138/mes** |

**Conclusion**: SageMaker es **caro para este lab**. El MLflow server de SageMaker cuesta $432/mes si corre 24/7 (Small instance) — equivalente a 6 EC2 t3.micro.

**Cuanto cuesta el free tier de SageMaker**:
- **250 horas** de ml.t3.medium notebooks (primeros 2 meses) — excelente para entrenar.
- **50 horas** de m4.xlarge o m5.xlarge training (primeros 2 meses) — perfecto para correr `dvc repro` 2-3 veces.
- **125 horas** de m4.xlarge o m5.xlarge Real-time Inference (primeros 2 meses) — tu API gratis por 5 dias corridos.
- **NO incluye MLflow Tracking Server** en free tier (a enero 2026, no aparece en la lista).

**Conclusion sobre SageMaker + Free Tier**: puedes usar SageMaker Studio + Training + Endpoints gratis los primeros 2 meses, **pero el MLflow server no es free tier** (cuesta $432/mes Small 24/7). **No tiene sentido usar SageMaker MLflow**; mejor self-hosted en EC2 t3.micro ($7.59/mes) y SageMaker solo para training/inference.

**Pros de Opcion C**:
- Servicios administrados (menos que mantener).
- SageMaker Studio es un IDE completo (notebooks, Git, debugger).
- SageMaker Pipelines = orquestacion nativa.
- SageMaker Model Monitor = drift detection administrado.

**Contras**:
- **Costo 5-10x mayor** que Opcion B.
- MLflow server de SageMaker no es free tier y es muy caro.
- Menos control del ambiente.
- Para este lab especifico (clasificación simple de 10 clases), es overkill.

### 13.3 Comparativa de las 3 opciones

| Dimension | Opcion A (Light) | Opcion B (Prod-Style) | Opcion C (SageMaker) |
|---|---|---|---|
| **Costo/mes post-free-tier** | ~$10 | ~$20-36 (con ALB) | ~$138 |
| **Meses cubiertos con $200 credits** | 16-25 | 5-10 | 1.4 (o gratis 2 meses para algunos) |
| **Servicios AWS** | 3 | 8-10 | 12+ |
| **Complejidad de setup** | Baja (1h) | Media (3-4h) | Alta (1-2 dias) |
| **MLflow** | En EC2 t3.micro ($7.59) | En EC2 t3.micro ($7.59) | SageMaker MLflow ($432/mes) — **NO recomendado** |
| **API serving** | Docker en EC2 o local | ECS Fargate serverless | SageMaker Endpoint |
| **Drift detection** | Script Python en EC2 | Lambda + CloudWatch | SageMaker Model Monitor |
| **CI/CD** | GitHub Actions | GitHub Actions + ECR | CodePipeline (opcional) |
| **Aspecto en portafolio** | Basico | Profesional | Senior (overkill) |
| **Recomendado para** | Hobby/demos rapidas | **Portfolio production-ready** | Empresas con presupuesto |

### 13.4 Mi recomendacion concreta para tu portafolio

**Ejecuta la Opcion B (Production-Style) con esta distribucion**:

1. **Mes 1-2 (free tier activo)**:
   - Levanta todo con la Opcion B usando los $200 credits.
   - Costo real: $0 de tu bolsillo.
   - Experimenta, itera, documenta.

2. **Mes 3-6 (cuando free tier expira o lo consumes)**:
   - **Apaga recursos cuando no los uses**: detener EC2 + ECS services cuando no estes mostrando el portafolio.
   - Costo si esta 24/7: ~$20/mes.
   - Costo si solo prendes para demos: ~$3-5/mes.

3. **Optimizaciones adicionales**:
   - Usar **Fargate Spot** (hasta 70% descuento): $20 → $6/mes.
   - Usar **S3 Intelligent-Tiering** (mueve data fria a Glacier automaticamente).
   - Configurar **Lifecycle policies** en S3 para borrar artifacts viejos.
   - Usar **CloudWatch Logs Insights** con retention de 7 dias (default es forever, $$$).

### 13.5 SageMaker Studio vs self-hosted EC2 — cuando usar cada uno

**Usa SageMaker Studio cuando**:
- Necesitas notebooks interactivos con kernels preconfigurados.
- Quieres correr experiments comparativos (SageMaker Experiments).
- El dataset es >10 GB y quieres notebooks con mas RAM (hasta ml.p4d.24xlarge).
- Quieres debugger visual de training (SageMaker Debugger con TensorBoard).
- Estas en una empresa y el CFO paga.

**Usa self-hosted (EC2 + MLflow) cuando**:
- Quieres control total del ambiente (puedes instalar lo que sea).
- El lab es pequeño/mediano (como este).
- MLflow es tu tracking system (no quieres SageMaker Experiments).
- Costo es una preocupacion ($7.59 vs $432/mes).
- Quieres mostrar habilidades de "MLOps engineer" (no solo "uso SageMaker").

**Para este portafolio especifico: self-hosted (Opcion B)**. Te dara mas talking points en entrevistas: "Configure el tracking de MLflow, automatice el reentrenamiento con ECS Fargate, use CloudWatch para alerting".

### 13.6 Drift detection en AWS (complemento a la seccion §5.3)

| Opcion | Servicio | Costo | Setup |
|---|---|---|---|
| Batch (manual) | Script Python en EC2 | $0 incremental | Correr `analyze_drift.py --mode batch` en cron |
| Batch (automatizado) | EventBridge + Lambda | ~$0.20/mes (1 corrida/dia) | EventBridge rule dispara Lambda que corre el script |
| Real-time (full) | SageMaker Model Monitor | $0.23/h instancia + $0.01/GB data | Solo si ya usas SageMaker Endpoints |
| Hibrido | CloudWatch Metric Streams + Alarms | $0.30/metrica/mes | Lambda emite metricas custom de drift; Alarm si > umbral |

**Para Opcion B**: usar **EventBridge + Lambda** para drift batch diario.

```yaml
# EventBridge rule (terraform-like)
DriftCheckSchedule:
  Type: AWS::Events::Rule
  Properties:
    ScheduleExpression: 'cron(0 6 * * ? *)'  # 06:00 UTC diario
    Targets:
      - Arn: !GetAtt DriftCheckFunction.Arn
        Id: drift-check-target

DriftCheckFunction:
  Type: AWS::Lambda::Function
  Properties:
    Runtime: python3.11
    Handler: handler.run
    Timeout: 300
    Environment:
      Variables:
        TRAIN_DATA: s3://mlops-dvc-data/train.csv
        PROD_DATA: s3://mlops-dvc-data/prod_sim.csv
```

### 13.7 Checklist de implementacion para Opcion B

**Pre-requisitos**:
- [ ] Cuenta AWS con $200 credits activos (nueva, primeros 6 meses).
- [ ] AWS CLI configurado (`aws configure`).
- [ ] Terraform o AWS CDK instalado (opcional, para IaC).
- [ ] Dominio propio (opcional, para Route53 + certificado SSL).

**Step-by-step** (resumido):

- [ ] **Paso 1**: Crear VPC, subnets, security groups (o usar default VPC para empezar).
- [ ] **Paso 2**: Crear 2 buckets S3: `mlops-dvc-data` y `mlops-mlflow-artifacts`.
- [ ] **Paso 3**: Lanzar EC2 t3.micro con Amazon Linux 2, instalar Python 3.11, MLflow, DVC.
- [ ] **Paso 4**: Correr MLflow server apuntando a S3 artifacts.
- [ ] **Paso 5**: Configurar security group de EC2: abrir 5000 a tu IP/32 (no 0.0.0.0/0).
- [ ] **Paso 6**: Crear ECR repository `spotify-api`.
- [ ] **Paso 7**: Build + push de la imagen Docker a ECR.
- [ ] **Paso 8**: Crear ECS cluster + task definition + service Fargate.
- [ ] **Paso 9**: Configurar CloudWatch Logs group `/ecs/spotify-api`.
- [ ] **Paso 10**: Probar `curl http://<Fargate-public-ip>:8000/health`.
- [ ] **Paso 11**: (Opcional) Crear ALB para HTTPS + dominio custom.
- [ ] **Paso 12**: (Opcional) Configurar EventBridge + Lambda para drift check diario.
- [ ] **Paso 13**: (Opcional) Crear dashboard CloudWatch con metricas del API.
- [ ] **Paso 14**: Documentar todo en un README de infra con diagrama de arquitectura.

### 13.8 Estimacion de tiempo y esfuerzo

| Opcion | Tiempo setup | Mantenimiento mensual | Valor portafolio |
|---|---|---|---|
| A (Light) | 1-2 horas | ~30 min | Bajo (se ve basico) |
| B (Prod-Style) | 4-6 horas | ~2 horas | **Alto (production-ready)** |
| C (SageMaker) | 1-2 dias | ~4 horas | Alto pero overkill |

### 13.9 Comandos utiles para el setup en AWS

```bash
# 1. Verificar credits disponibles
aws budgets describe-budgets --account-id $(aws sts get-caller-identity --query Account --output text)

# 2. Crear recursos base
aws s3 mb s3://mlops-spotify-portfolio-dvc
aws s3 mb s3://mlops-spotify-portfolio-mlflow
aws ecr create-repository --repository-name spotify-api

# 3. Lanzar EC2 con UserData (instala MLflow automaticamente)
aws ec2 run-instances \
  --image-id ami-0c02fb55956d7a4f6 \
  --instance-type t3.micro \
  --user-data '#!/bin/bash
    yum install -y python3 git
    pip3 install mlflow boto3 dvc[s3]
    nohup mlflow server --host 0.0.0.0 --port 5000 \
      --default-artifact-root s3://mlops-spotify-portfolio-mlflow/artifacts \
      --backend-store-uri sqlite:///mlflow.db > /var/log/mlflow.log 2>&1 &' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mlflow-server}]'

# 4. Build + push Docker a ECR
$(aws ecr get-login --region us-east-1 --no-include-email)
docker build -t spotify-api ./model_serving
docker tag spotify-api:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/spotify-api:latest
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/spotify-api:latest

# 5. Deploy a ECS Fargate
aws ecs create-cluster --cluster-name mlops-cluster
# (registrar task definition + crear service, ver seccion 13.4)

# 6. Cleanup cuando no uses (importante para controlar costos!)
aws ecs update-service --cluster mlops-cluster --service spotify-api --desired-count 0
aws ec2 stop-instances --instance-ids i-xxxxx
```

### 13.10 Consideraciones de seguridad (importante en AWS)

| Tema | Recomendacion |
|---|---|
| **Puertos abiertos** | NUNCA `0.0.0.0/0` en production. Usar tu IP/32 o VPN. |
| **Credenciales AWS** | Usar IAM Roles para EC2/ECS, no access keys. |
| **MLflow sin auth** | MLflow open-source NO tiene autenticacion. Usar SSH tunnel o ALB con Cognito. |
| **Secrets** | AWS Secrets Manager para DB passwords, API keys. NUNCA en .env files en EC2. |
| **S3 public access** | Bloquear con `BlockPublicAccess` a nivel de cuenta. |
| **Logs sensibles** | NO loguear audio features raw a CloudWatch (contienen info inferible). Usar metricas agregadas. |
| **Imagen Docker** | Escanear con Trivy/Snyk (CI/CD del Anexo A.6). |
| **Data en S3** | Encriptar en reposo (SSE-S3 default) y en transito (HTTPS only). |

### 13.11 Proximos pasos concretos

Si decides ir con **Opcion B**, te sugiero este roadmap:

| Semana | Actividad | Deliverable |
|---|---|---|
| 1 | Implementar TODOs locales; verificar 20/20 | PR verde, lab completo |
| 2 | Setup AWS: EC2 + S3 + ECR + ECS Fargate | Stack corriendo en AWS |
| 3 | Adaptar el Dockerfile para ECR (build args, ECR login) | `docker push` funciona |
| 4 | Agregar OpenTelemetry + CloudWatch integration (seccion §11) | Dashboard de metricas |
| 5 | Crear Lambda + EventBridge para drift check diario | Job automatizado de drift |
| 6 | Documentar todo en un repo de infra (Terraform/CDK o manual) | README con arquitectura + costos |

**Total: 6 semanas para un portafolio production-ready que impresione en entrevistas.**

---

## Notas Finales

- Este documento es **vivo**: actualizalo conforme avanzas.
- Antes del PR final, revisa el `GRADING_RUBRIC.md` y marca cada item del Submission Checklist.
- El PR debe pasar CI en **verde** para obtener el punto extra de la rubrica 4.3.
- Si tienes dudas sobre un error especifico, busca primero en la seccion "Troubleshooting" del README.

**Exito con la entrega!**
