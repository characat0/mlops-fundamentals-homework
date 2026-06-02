import pytest
import os
import pandas as pd
import tempfile
from src.process import process_data


def test_process_data_temporal_split():
    """Test that process_data correctly splits data temporally by year."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.csv")
        train_output = os.path.join(tmpdir, "train.csv")
        prod_output = os.path.join(tmpdir, "prod.csv")

        # Create sample Spotify data with proper headers
        sample_data = pd.DataFrame({
            "year": [2005, 2008, 2010, 2012, 2015],
            "genre": ["Rock", "Pop", "Rock", "Electronic", "Hip-Hop"],
            "danceability": [0.5, 0.6, 0.7, 0.8, 0.9],
            "energy": [0.1, 0.2, 0.3, 0.4, 0.5],
            "tempo": [100, 110, 120, 130, 140],
        })
        sample_data.to_csv(input_path, index=False)

        process_data(input_path, train_output, prod_output, year_threshold=2010)

        assert os.path.exists(train_output), "Training data file not created"
        assert os.path.exists(prod_output), "Production data file not created"

        train_df = pd.read_csv(train_output)
        prod_df = pd.read_csv(prod_output)

        assert len(train_df) == 3, f"Expected 3 training records (year <= 2010), got {len(train_df)}"
        assert len(prod_df) == 2, f"Expected 2 production records (year > 2010), got {len(prod_df)}"
        assert (train_df["year"] <= 2010).all(), "Training data has years > 2010"
        assert (prod_df["year"] > 2010).all(), "Production data has years <= 2010"


def test_process_data_preserves_features():
    """Test that process_data preserves all audio features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.csv")
        train_output = os.path.join(tmpdir, "train.csv")
        prod_output = os.path.join(tmpdir, "prod.csv")

        # Create sample with multiple features
        sample_data = pd.DataFrame({
            "year": [2000, 2010],
            "genre": ["Rock", "Pop"],
            "danceability": [0.1, 0.2],
            "energy": [0.3, 0.4],
            "tempo": [0.5, 0.6],
        })
        sample_data.to_csv(input_path, index=False)

        process_data(input_path, train_output, prod_output)

        train_df = pd.read_csv(train_output)
        # Should preserve all columns from input
        assert len(train_df.columns) == 5, f"Expected 5 columns, got {len(train_df.columns)}"
        assert "year" in train_df.columns, "Missing 'year' column"
        assert "genre" in train_df.columns, "Missing 'genre' column"
        assert "danceability" in train_df.columns, "Missing audio features"
