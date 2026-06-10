from pathlib import Path
import shutil

import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier


FEATURE_NAMES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]


def main():
    X = pd.DataFrame(
        [
            {
                "danceability": 0.7,
                "energy": 0.8,
                "key": 5,
                "loudness": -5.0,
                "mode": 1,
                "speechiness": 0.05,
                "acousticness": 0.1,
                "instrumentalness": 0.0,
                "liveness": 0.2,
                "valence": 0.6,
                "tempo": 120.0,
                "duration_ms": 240000,
            }
        ],
        columns=FEATURE_NAMES,
    )

    y = [7]

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    model_path = Path("model_serving") / "models"

    if model_path.exists():
        shutil.rmtree(model_path)

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(model_path),
        input_example=X,
    )


if __name__ == "__main__":
    main()