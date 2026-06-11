import argparse
import logging
import os

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(source_path: str, output_path: str) -> None:
    """Load the Kaggle Spotify songs CSV and save it as raw data."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Loading Spotify songs from %s", source_path)
    df = pd.read_csv(source_path)

    logger.info("Raw dataset shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))

    df.to_csv(output_path, index=False)
    logger.info("Saved raw data to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    load_data(args.source_path, args.output_path)
