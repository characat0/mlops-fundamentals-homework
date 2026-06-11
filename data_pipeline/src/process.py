import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data(
    input_path: str,
    train_output: str,
    prod_output: str,
    year_threshold: int = 2010
):
    """
    Load and split the Spotify dataset temporally by release year.

    Creates a temporal train/prod split that simulates real-world data drift:
    - train (year <= threshold): pre-streaming era (CD/iTunes)
    - prod  (year >  threshold): streaming era (Spotify/Apple Music)

    Args:
        input_path: Path to raw dataset CSV (from load.py)
        train_output: Path to save training split (year <= year_threshold)
        prod_output: Path to save production split (year > year_threshold)
        year_threshold: Year boundary (default 2010)
    """
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Year range: {df['year'].min()}-{df['year'].max()}")

    train_df = df[df["year"] <= year_threshold].copy()
    prod_df = df[df["year"] > year_threshold].copy()

    logger.info(f"Training split shape: {train_df.shape}")
    logger.info(f"Production simulation split shape: {prod_df.shape}")

    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(prod_output), exist_ok=True)

    train_df.to_csv(train_output, index=False)
    prod_df.to_csv(prod_output, index=False)

    logger.info(f"Saved training data to {train_output}")
    logger.info(f"Saved production simulation data to {prod_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--prod_output", type=str, required=True)
    parser.add_argument("--year_threshold", type=int, default=2010)
    args = parser.parse_args()

    process_data(
        args.input_path,
        args.train_output,
        args.prod_output,
        args.year_threshold
    )
