import os
import pandas as pd
import tempfile
from src.process import process_data


def test_process_data_temporal_split():
    """Test that process_data correctly splits data temporally by year with year_threshold=2010."""
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
            "key": [1, 2, 3, 4, 5],
            "loudness": [-5.0, -4.5, -4.0, -3.5, -3.0],
            "mode": [0, 1, 0, 1, 0],
            "speechiness": [0.1, 0.15, 0.2, 0.25, 0.3],
            "acousticness": [0.9, 0.8, 0.7, 0.6, 0.5],
            "instrumentalness": [0.01, 0.02, 0.03, 0.04, 0.05],
            "liveness": [0.1, 0.2, 0.3, 0.4, 0.5],
            "valence": [0.5, 0.6, 0.7, 0.8, 0.9],
            "tempo": [100, 110, 120, 130, 140],
            "duration_ms": [180000, 190000, 200000, 210000, 220000],
        })
        sample_data.to_csv(input_path, index=False)

        process_data(input_path, train_output, prod_output, year_threshold=2010)

        assert os.path.exists(train_output), "Training data file not created"
        assert os.path.exists(prod_output), "Production data file not created"

        train_df = pd.read_csv(train_output)
        prod_df = pd.read_csv(prod_output)

        msg_train = f"Expected 3 training records (year <= 2010), got {len(train_df)}"
        assert len(train_df) == 3, msg_train
        msg_prod = f"Expected 2 production records (year > 2010), got {len(prod_df)}"
        assert len(prod_df) == 2, msg_prod
        assert (train_df["year"] <= 2010).all(), "Training data has years > 2010"
        assert (prod_df["year"] > 2010).all(), "Production data has years <= 2010"


def test_process_data_preserves_audio_features():
    """Test that process_data preserves all expected audio feature columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.csv")
        train_output = os.path.join(tmpdir, "train.csv")
        prod_output = os.path.join(tmpdir, "prod.csv")

        # Create sample with all expected audio features
        expected_audio_features = [
            "danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
        ]

        sample_data = pd.DataFrame({
            "year": [2005, 2015],
            "genre": ["Rock", "Pop"],
            "danceability": [0.1, 0.2],
            "energy": [0.3, 0.4],
            "key": [1, 2],
            "loudness": [-5.0, -4.5],
            "mode": [0, 1],
            "speechiness": [0.1, 0.15],
            "acousticness": [0.9, 0.8],
            "instrumentalness": [0.01, 0.02],
            "liveness": [0.1, 0.2],
            "valence": [0.5, 0.6],
            "tempo": [100, 110],
            "duration_ms": [180000, 190000],
        })
        sample_data.to_csv(input_path, index=False)

        process_data(input_path, train_output, prod_output, year_threshold=2010)

        train_df = pd.read_csv(train_output)
        prod_df = pd.read_csv(prod_output)

        # Validate all expected audio feature columns are present in both splits
        for feature in expected_audio_features:
            msg_miss_train = f"Missing audio feature '{feature}' in training data"
            assert feature in train_df.columns, msg_miss_train
            msg_miss_prod = f"Missing audio feature '{feature}' in production data"
            assert feature in prod_df.columns, msg_miss_prod


def test_process_data_year_boundary_condition():
    """Test year boundary condition: records with year=2010 go to train, year=2011 go to prod."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.csv")
        train_output = os.path.join(tmpdir, "train.csv")
        prod_output = os.path.join(tmpdir, "prod.csv")

        # Create sample data with exact boundary years
        sample_data = pd.DataFrame({
            "year": [2009, 2010, 2011, 2012],
            "genre": ["Rock", "Pop", "Jazz", "Electronic"],
            "danceability": [0.5, 0.6, 0.7, 0.8],
            "energy": [0.1, 0.2, 0.3, 0.4],
            "key": [1, 2, 3, 4],
            "loudness": [-5.0, -4.5, -4.0, -3.5],
            "mode": [0, 1, 0, 1],
            "speechiness": [0.1, 0.15, 0.2, 0.25],
            "acousticness": [0.9, 0.8, 0.7, 0.6],
            "instrumentalness": [0.01, 0.02, 0.03, 0.04],
            "liveness": [0.1, 0.2, 0.3, 0.4],
            "valence": [0.5, 0.6, 0.7, 0.8],
            "tempo": [100, 110, 120, 130],
            "duration_ms": [180000, 190000, 200000, 210000],
        })
        sample_data.to_csv(input_path, index=False)

        process_data(input_path, train_output, prod_output, year_threshold=2010)

        train_df = pd.read_csv(train_output)
        prod_df = pd.read_csv(prod_output)

        # Verify boundary: 2010 goes to train, 2011+ go to prod
        msg_err_train = f"Expected 2 training records (2009, 2010), got {len(train_df)}"
        assert len(train_df) == 2, msg_err_train
        msg_err_prod = f"Expected 2 production records (2011, 2012), got {len(prod_df)}"
        assert len(prod_df) == 2, msg_err_prod

        # Verify exact years in each split
        set_train = set(train_df["year"])
        msg_train = f"Expected years {{2009, 2010}} in train, got {set_train}"
        assert set_train == {2009, 2010}, msg_train
        set_prod = set(prod_df["year"])
        msg_prod = f"Expected years {{2011, 2012}} in prod, got {set_prod}"
        assert set_prod == {2011, 2012}, msg_prod

        # Verify threshold boundary
        assert (train_df["year"] <= 2010).all(), "Training data contains years > 2010"
        assert (prod_df["year"] > 2010).all(), "Production data contains years <= 2010"
