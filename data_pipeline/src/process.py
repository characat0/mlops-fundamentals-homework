import argparse
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data(
    input_path: str, train_output: str, prod_output: str, year_threshold: int = 2010
):
    """
    Load and split the Spotify dataset temporally by release year.

    Creates a temporal train/prod split that simulates real-world data drift:
    - train (year <= threshold): pre-streaming era (CD/iTunes)
    - prod  (year >  threshold): streaming era (Spotify/Apple Music)

    The 2010 boundary marks Spotify's launch. Audio feature distributions shift
    significantly across this boundary — this is intentional, it's the drift
    students will detect in analyze_drift.py.

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

    # TODO: Split df into two DataFrames using boolean indexing on the 'year' column:
    #   train_df — rows where year <= year_threshold
    train_df = df[df["year"] <= year_threshold]
    #   prod_df  — rows where year >  year_threshold
    prod_df = df[df["year"] > year_threshold]
    #
    # Log the size of each split so you can sanity-check the ratio.
    logger.info(f"Training dataset shape: {train_df.shape}")
    logger.info(f"Production dataset shape: {prod_df.shape}")

    # TODO: Save both splits to CSV (index=False).
    #   Create parent directories first with os.makedirs(..., exist_ok=True).
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(prod_output), exist_ok=True)
    #   train_df → train_output
    train_df.to_csv(train_output, index=False)
    #   prod_df  → prod_output
    prod_df.to_csv(prod_output, index=False)

    logger.info(f"Saved training dataset to {train_output}")
    logger.info(f"Saved production dataset to {prod_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--prod_output", type=str, required=True)
    parser.add_argument("--year_threshold", type=int, default=2010)
    args = parser.parse_args()

    process_data(
        args.input_path, args.train_output, args.prod_output, args.year_threshold
    )
