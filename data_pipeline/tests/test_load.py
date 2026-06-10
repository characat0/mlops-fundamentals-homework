import os
import tempfile
import pandas as pd
from src.load import load_data


def test_load_data_creates_output():
    """Test that load_data creates the output CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "songs.csv")
        output_path = os.path.join(tmpdir, "data", "raw.csv")
        sample = pd.DataFrame(
            {
                "year": [2005, 2015],
                "genre": ["Rock", "Pop"],
                "danceability": [0.5, 0.7],
            }
        )
        sample.to_csv(input_path, index=False)
        load_data(input_path, output_path)
        assert os.path.exists(output_path)


def test_load_data_preserves_all_columns():
    """Test that load_data keeps all columns from the source CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "songs.csv")
        output_path = os.path.join(tmpdir, "data", "raw.csv")
        sample = pd.DataFrame(
            {
                "year": [2005],
                "genre": ["Rock"],
                "danceability": [0.5],
                "energy": [0.8],
                "extra_column": ["metadata"],
            }
        )
        sample.to_csv(input_path, index=False)
        load_data(input_path, output_path)
        result = pd.read_csv(output_path)
        assert list(result.columns) == list(sample.columns)


def test_load_data_row_count_unchanged():
    """Test that load_data preserves all rows from the source CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "songs.csv")
        output_path = os.path.join(tmpdir, "data", "raw.csv")
        sample = pd.DataFrame(
            {
                "year": [2005, 2010, 2015],
                "genre": ["Rock", "Pop", "Jazz"],
                "danceability": [0.5, 0.6, 0.7],
            }
        )
        sample.to_csv(input_path, index=False)
        load_data(input_path, output_path)
        result = pd.read_csv(output_path)
        assert len(result) == 3
