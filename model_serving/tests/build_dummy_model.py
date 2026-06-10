"""
Build a minimal MLflow model so the API can be tested without the
real @champion model being present.

Usage:
    uv run python tests/build_dummy_model.py [target_dir]

The default target_dir is 'models/' (where the API looks for
artifacts at startup). This is meant for CI and local development
where the real champion has not been baked in.

The shape of the artifacts (file names + format) matches what
data_pipeline/src/train.py produces, so the same
`mlflow.pyfunc.load_model()` call in app.main works in both cases.
The dummy model is a LogisticRegression that always predicts class
0 (the first genre alphabetically, which is 'Blues' given our
LabelEncoder training). It is enough to satisfy the
predict/predict_proba contract — the test only checks that the
endpoint returns 200 with a 'genre' and 'confidence' key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]
GENRES = [
    "Blues", "Classical", "Country", "Electronic", "Folk",
    "Hip-Hop", "Jazz", "Pop", "R&B", "Rock",
]


def build_dummy(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(GENRES)
    joblib.dump(label_encoder, models_dir / "label_encoder.joblib")

    scaler = StandardScaler()
    scaler.fit(np.zeros((1, len(AUDIO_FEATURES))))
    joblib.dump(scaler, models_dir / "scaler.joblib")

    joblib.dump(AUDIO_FEATURES, models_dir / "feature_order.joblib")

    model = LogisticRegression(max_iter=1000, C=1.0)
    X_dummy = np.zeros((2, len(AUDIO_FEATURES)))
    y_dummy = np.array([0, 1])
    model.fit(X_dummy, y_dummy)

    tmp_dir = models_dir.parent / "_dummy_mlflow_staging"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    mlflow.sklearn.save_model(model, path=tmp_dir.resolve().as_posix())

    for f in tmp_dir.iterdir():
        target = models_dir / f.name
        if f.is_file():
            target.write_bytes(f.read_bytes())
    import shutil
    shutil.rmtree(tmp_dir)

    print(f"Dummy model written to {models_dir}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models")
    build_dummy(target)
