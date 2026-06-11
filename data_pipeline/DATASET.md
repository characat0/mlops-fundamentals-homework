# Dataset Setup

The dataset is intentionally not committed to this repository.

1. Keep your Kaggle credentials outside the repo at:

   ```text
   C:\Users\rodri\.kaggle\kaggle.json
   ```

2. Download and unzip the Spotify songs dataset:

   ```bash
   kaggle datasets download -d serkantysz/550k-spotify-songs-audio-lyrics-and-genres
   ```

3. Place the extracted CSV at:

   ```text
   data_pipeline/songs.csv
   ```

4. Run the data pipeline from `data_pipeline/`:

   ```bash
   dvc repro
   ```

Do not commit `kaggle.json`, `songs.csv`, `data/*.csv`, MLflow artifacts, models, or API logs.
