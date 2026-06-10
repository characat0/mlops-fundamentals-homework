"""
Pytest configuration for model_serving.

If the champion model artifacts are missing from both models/remote/
and models/local/, build a minimal dummy model so the test suite can
run in any environment (CI, fresh clone, dev workstation without MLflow).

The dummy is functionally trivial (always predicts class 0) but it
exercises the same code paths as the real champion: MLflow pyfunc
loading, LabelEncoder decoding, scaler application, predict_proba.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BASE_DIR = Path(__file__).resolve().parent.parent
REMOTE_MODELS = _BASE_DIR / "models" / "remote"
LOCAL_MODELS = _BASE_DIR / "models" / "local"


def _ensure_dummy() -> None:
    # If either source already has a valid model, nothing to do.
    if (REMOTE_MODELS / "model.ubj").is_file():
        return
    if (LOCAL_MODELS / "model.ubj").is_file() and (LOCAL_MODELS / "label_encoder.joblib").is_file():
        return

    # No model available anywhere — build a dummy into models/local/
    # so the API can load it during tests.
    from tests.build_dummy_model import build_dummy

    build_dummy(LOCAL_MODELS)


@pytest.fixture(scope="session", autouse=True)
def _prepare_model() -> None:
    _ensure_dummy()
    yield
