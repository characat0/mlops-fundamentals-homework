# Data Pipeline

DVC orchestrated pipeline: Load → process → train → evaluate.

## Dataset Verification

`songs.csv.dvc` records the expected MD5 of the correct Kaggle dataset. After placing your `songs.csv` in this directory, verify it before running the pipeline:

```bash
# Check your file matches the expected hash
dvc status songs.csv.dvc
```

- **Clean** (no output / "Data and pipelines are up to date") → correct file
- **Modified** or **not in cache** → wrong file, re-download from Kaggle

**Expected MD5:** `0e71e2c46244acac485bd8c245aa6e56`

To manually check on macOS/Linux:
```bash
md5sum songs.csv        # Linux
md5 -q songs.csv        # macOS
```

## Pipeline Stages

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| `load` | `src/load.py` | `songs.csv` | `data/raw.csv` |
| `process` | `src/process.py` | `data/raw.csv` | `data/train.csv`, `data/prod_sim.csv` |
| `train` | `src/train.py` | `data/train.csv` | `models/` |
| `evaluate` | `src/evaluate.py` | `models/` | `metrics.json` |

## Running

```bash
dvc repro          # Run full pipeline
dvc dag            # Visualise the DAG
dvc status         # Check what needs to rerun
```

## Testing

```bash
pytest tests/
```
