# Procedimiento de Testeo del Stage 1: Data Pipeline

Sigue estos pasos para validar que la pipeline de datos cumple con la rúbrica de evaluación y el README.

## 1. Configuración Inicial (Reset Limpio)
Si has tenido problemas con registros previos o quieres asegurar una ejecución desde cero, ejecuta estos comandos en la raíz del proyecto utilizando **PowerShell**:

```powershell
# 1. Detener procesos previos de MLflow o Python
Stop-Process -Name "python", "mlflow" -Force -ErrorAction SilentlyContinue

# 2. Limpiar archivos de estado y bases de datos previas
Remove-Item -Recurse -Force .dvc, dvc.lock, mlflow.db, mlruns, models, metrics.json -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Filter "mlruns" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "models" | Remove-Item -Recurse -Force

# 3. Inicializar DVC (si no lo está)
uv run dvc init --no-scm
```

## 2. Preparación del Dataset (Rubric §1.1)
Coloca el archivo `songs.csv` descargado de Kaggle dentro de la carpeta `data_pipeline/`.

```powershell
cd data_pipeline
# Confirmar integridad: el hash debe coincidir con songs.csv.dvc
uv run dvc status songs.csv.dvc

# Si el archivo es nuevo para DVC ("not in cache"), confírmalo:
uv run dvc commit songs.csv.dvc
```

## 3. Ejecución de la Pipeline (Rubric §1.5)
Para evitar múltiples bases de datos en Windows, define una ruta absoluta para el tracking de MLflow:

```powershell
# Ejecutar desde la RAÍZ del proyecto
$env:MLFLOW_TRACKING_URI = "sqlite:///$((Get-Location).Path)\mlflow.db"

cd data_pipeline
uv run dvc repro
```

## 4. Verificación de Resultados

### A. Split Temporal (Rubric §1.2)
Verifica que la lógica de `process.py` respetó el límite del año 2010 (usando comillas dobles escapadas para PowerShell):

```powershell
# Verificar años en Train (debe ser <= 2010)
uv run python -c "import pandas as pd; df=pd.read_csv('data_pipeline/data/train.csv'); print(f'Max Year Train: {df.year.max()}')"

# Verificar años en Prod (debe ser > 2010)
uv run python -c "import pandas as pd; df=pd.read_csv('data_pipeline/data/prod_sim.csv'); print(f'Min Year Prod: {df.year.min()}')"
```

### B. MLflow UI y Registro (Rubric §1.3 y §1.4)
Inicia el servidor para inspeccionar el Model Registry:

```powershell
# Ejecutar desde la RAÍZ del proyecto
$env:MLFLOW_TRACKING_URI = "sqlite:///$((Get-Location).Path)\mlflow.db"
uv run mlflow server --host 127.0.0.1 --port 5000
```
Visita `http://localhost:5000` en tu navegador:
1.  **Experiments**: Debes ver runs para `logistic_regression` y `xgboost`.
2.  **Models**: Debe aparecer **`spotify-genre-classifier`**. (Si ves otros nombres de pruebas anteriores, puedes ignorarlos).
3.  **Aliases**: Entra al modelo y verifica que una versión tenga el tag **`champion`**.

### C. Tests Unitarios y Estilo (Rubric §4)
Asegúrate de instalar `flake8` y configurar el path correctamente:

```powershell
# Instalar dependencias faltantes
uv add flake8

# Ejecutar tests unitarios desde la RAÍZ
$env:PYTHONPATH = "data_pipeline"
uv run pytest data_pipeline/tests/ -v

# Verificar estilo de código (enfocarse en src)
uv run flake8 data_pipeline/src/
```

Si todos estos pasos se completan con éxito, el Stage 1 cumple con el 100% de los requisitos.
