import argparse
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(source_path: str, output_path: str):
    """
    Load the Kaggle Spotify songs CSV and save as raw data.
    Responsibility: reads the raw Kaggle CSV and saves it as-is.
    Column filtering and temporal splitting happen in process.py, not here.
    Args:
        source_path: Path to songs.csv (downloaded from Kaggle)
        output_path: Path where raw.csv should be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Loading Spotify songs from {source_path}...")
    df = pd.read_csv(source_path)
    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    # All columns are kept here — filtering happens in process.py
    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    load_data(args.source_path, args.output_path)
