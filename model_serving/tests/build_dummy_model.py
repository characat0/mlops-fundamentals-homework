"""
Build a minimal XGBoost model so the API can be tested without the
real @champion model being present.

Usage:
    uv run python tests/build_dummy_model.py [target_dir]

The default target_dir is 'models/local/' (where the API looks for
artifacts at startup). This is meant for CI and local development
where the real champion has not been baked in.

The dummy model is an XGBClassifier trained on 2 samples. It is
enough to satisfy the predict/predict_proba contract — the test only
checks that the endpoint returns 200 with a 'genre' and 'confidence'
key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
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

    # Train a minimal XGBClassifier and save as .ubj
    model = xgb.XGBClassifier(
        n_estimators=2,
        max_depth=1,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    X_dummy = np.random.rand(20, len(AUDIO_FEATURES))
    y_dummy = np.arange(20) % len(GENRES)  # 0-9 cycling
    model.fit(X_dummy, y_dummy)
    model.save_model(str(models_dir / "model.ubj"))

    print(f"Dummy model written to {models_dir}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/local")
    build_dummy(target)
