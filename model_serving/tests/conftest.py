import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def make_mock_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([7])  # índice → 'Pop'
    mock_model.predict_proba.return_value = np.array([
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.85, 0.04, 0.04]
    ])
    return mock_model


@pytest.fixture(autouse=True)
def mock_mlflow_model():
    import app.main as main_module
    mock_model = make_mock_model()
    original = main_module._model
    main_module._model = mock_model
    yield
    main_module._model = original
