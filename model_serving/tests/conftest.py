"""
Pytest configuration for model_serving.

If the champion model artifacts are missing from ./models, build a
minimal dummy model in place so the test suite can run in any
environment (CI, fresh clone, dev workstation without MLflow access).

The dummy is functionally trivial (always predicts class 0) but it
exercises the same code paths as the real champion: MLflow pyfunc
loading, LabelEncoder decoding, scaler application, predict_proba.
"""
from __future__ import annotations

from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PREPROCESSORS_DIR = Path(__file__).resolve().parent.parent / "preprocessors"


def _ensure_dummy() -> None:
    if (MODELS_DIR / "MLmodel").is_file():
        return
    if (MODELS_DIR / "label_encoder.joblib").is_file():
        return

    if PREPROCESSORS_DIR.is_dir():
        for joblib_file in PREPROCESSORS_DIR.glob("*.joblib"):
            target = MODELS_DIR / joblib_file.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(joblib_file.read_bytes())

    if (MODELS_DIR / "MLmodel").is_file():
        return

    from tests.build_dummy_model import build_dummy

    build_dummy(MODELS_DIR)


@pytest.fixture(scope="session", autouse=True)
def _prepare_model() -> None:
    _ensure_dummy()
    yield
