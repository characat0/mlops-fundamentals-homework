import argparse
import json
import logging

import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms"
]


def run_ks_analysis(train_df: pd.DataFrame, prod_df: pd.DataFrame, output_path: str) -> dict:
    """
    Run a Kolmogorov-Smirnov test for each audio feature.

    Args:
        train_df: DataFrame with training feature distributions (baseline)
        prod_df: DataFrame with production feature distributions (to compare)
        output_path: Where to write drift_report.json

    Returns:
        dict with per-feature KS results and an overall status
    """
    logger.info(f"Training samples: {len(train_df)} | Production samples: {len(prod_df)}")

    drift_results = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "train_samples": len(train_df),
        "production_samples": len(prod_df),
        "features_with_drift": 0,
        "drifted_features": [],
        "details": {}
    }

    features_to_test = [
        feature
        for feature in AUDIO_FEATURES
        if feature in train_df.columns and feature in prod_df.columns
    ]

    for feature in features_to_test:
        train_values = train_df[feature].dropna()
        prod_values = prod_df[feature].dropna()

        ks_statistic, p_value = stats.ks_2samp(train_values, prod_values)
        drift_detected = p_value < 0.05

        drift_results["details"][feature] = {
            "ks_statistic": float(ks_statistic),
            "p_value": float(p_value),
            "drift_detected": bool(drift_detected),
            "train_mean": float(train_values.mean()),
            "prod_mean": float(prod_values.mean())
        }

        if drift_detected:
            drift_results["features_with_drift"] += 1
            drift_results["drifted_features"].append(feature)

    drifted = drift_results["features_with_drift"]
    total = len(features_to_test)
    drift_results["drift_percentage"] = (drifted / total * 100) if total > 0 else 0
    if drift_results["drift_percentage"] > 20:
        drift_results["status"] = "DRIFT_DETECTED"
    else:
        drift_results["status"] = "NORMAL"

    logger.info(f"Status: {drift_results['status']} "
                f"({drifted}/{total} features drifted)")

    with open(output_path, "w") as f:
        json.dump(drift_results, f, indent=2)

    logger.info(f"Drift report saved to {output_path}")
    return drift_results


def analyze_batch_drift(train_path: str, prod_path: str, output_path: str) -> dict:
    """
    Batch drift analysis: compare train.csv vs prod_sim.csv.

    Args:
        train_path: Path to data/train.csv (output of process.py)
        prod_path: Path to prod_sim.csv (output of process.py)
        output_path: Where to write drift_report.json
    """
    logger.info(f"[BATCH] Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    logger.info(f"[BATCH] Loading production data from {prod_path}")
    prod_df = pd.read_csv(prod_path)

    return run_ks_analysis(train_df, prod_df, output_path)


def analyze_online_drift(train_path: str, api_logs_path: str, output_path: str) -> dict:
    """
    Online drift analysis: compare train.csv vs live API request logs.

    Args:
        train_path: Path to data/train.csv (baseline distribution)
        api_logs_path: Path to logs/api_requests.jsonl (from FastAPI middleware)
        output_path: Where to write drift_report.json
    """
    logger.info(f"[ONLINE] Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    logger.info(f"[ONLINE] Loading API logs from {api_logs_path}")
    try:
        api_logs = []
        with open(api_logs_path, "r") as f:
            for line in f:
                if line.strip():
                    api_logs.append(json.loads(line))
        api_df = pd.DataFrame(api_logs)
    except FileNotFoundError:
        logger.warning("API logs not found. Run the API and make some predictions first.")
        return {"status": "no_api_logs", "message": "No API requests logged yet"}

    if api_df.empty:
        return {"status": "no_api_logs", "message": "API logs are empty"}

    return run_ks_analysis(train_df, api_df, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift detection for Spotify genre classifier")
    parser.add_argument(
        "--mode",
        choices=["batch", "online"],
        required=True,
        help=(
            "batch: compare data/train.csv vs data/prod_sim.csv; "
            "online: compare data/train.csv vs logs/api_requests.jsonl"
        )
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training CSV (data/train.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="drift_report.json",
        help="Path to write the drift report JSON"
    )
    parser.add_argument(
        "--prod_data",
        type=str,
        help="[batch mode] Path to prod_sim.csv"
    )
    parser.add_argument(
        "--api_logs",
        type=str,
        help="[online mode] Path to api_requests.jsonl"
    )

    args = parser.parse_args()

    if args.mode == "batch":
        if not args.prod_data:
            parser.error("--prod_data is required for batch mode")
        analyze_batch_drift(args.train_data, args.prod_data, args.output)

    elif args.mode == "online":
        if not args.api_logs:
            parser.error("--api_logs is required for online mode")
        analyze_online_drift(args.train_data, args.api_logs, args.output)
