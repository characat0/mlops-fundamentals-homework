import argparse
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data(
    input_path: str,
    train_output: str,
    prod_output: str,
    year_threshold: int = 2005
):
    """
    Load and split the Spotify dataset temporally by release year.

    **Responsibility**: Load raw CSV, filter/select columns, and split into train/prod.
    This creates a temporal train/test split to simulate real-world data drift.

    Args:
        input_path: Path to raw dataset CSV (from download.py)
        train_output: Path to save training data (year <= threshold)
        prod_output: Path to save production data (year > threshold)
        year_threshold: Year to split on (default 2005)

    TODO:
        1. Load CSV with headers
        2. Select relevant columns:
           - Target: genre
           - Features: danceability, energy, key, loudness, mode, speechiness,
                       acousticness, instrumentalness, liveness, valence, tempo, duration_ms
           - Temporal: year
        3. Split temporally: train (year <= threshold), prod_sim (year > threshold)
        4. Save both subsets as CSV
    """
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Year range: {df['year'].min()}-{df['year'].max()}")

    # Temporal split
    train_df = df[df["year"] <= year_threshold]
    prod_df = df[df["year"] > year_threshold]

    logger.info(f"Training set (year <= {year_threshold}): {len(train_df)} records")
    logger.info(f"Production set (year > {year_threshold}): {len(prod_df)} records")

    train_df.to_csv(train_output, index=False)
    prod_df.to_csv(prod_output, index=False)

    logger.info(f"Saved training data to {train_output}")
    logger.info(f"Saved production data to {prod_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--prod_output", type=str, required=True)
    parser.add_argument("--year_threshold", type=int, default=2005)
    args = parser.parse_args()

    process_data(
        args.input_path,
        args.train_output,
        args.prod_output,
        args.year_threshold
    )
