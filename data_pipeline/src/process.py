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
    year_threshold: int = 2010,
) -> None:
    """Split Spotify data into temporal train/prod datasets."""
    logger.info("Loading data from %s", input_path)
    df = pd.read_csv(input_path)

    if "year" not in df.columns:
        raise ValueError("Input data must contain a 'year' column.")

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    logger.info("Raw dataset shape: %s", df.shape)
    logger.info("Year range: %s-%s", df["year"].min(), df["year"].max())

    train_df = df[df["year"] <= year_threshold].copy()
    prod_df = df[df["year"] > year_threshold].copy()

    logger.info("Train split shape: %s", train_df.shape)
    logger.info("Production simulation split shape: %s", prod_df.shape)

    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(prod_output), exist_ok=True)

    train_df.to_csv(train_output, index=False)
    prod_df.to_csv(prod_output, index=False)

    logger.info("Saved train data to %s", train_output)
    logger.info("Saved prod simulation data to %s", prod_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--prod_output", type=str, required=True)
    parser.add_argument("--year_threshold", type=int, default=2010)
    args = parser.parse_args()

    process_data(
        input_path=args.input_path,
        train_output=args.train_output,
        prod_output=args.prod_output,
        year_threshold=args.year_threshold,
    )
