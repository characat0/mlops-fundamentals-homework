import argparse
import pandas as pd
import numpy as np
import json
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_drift(train_data_path: str, api_logs_path: str, output_path: str = "drift_report.json"):
    """
    Compare statistical distributions of training data vs production logs.

    Detects data drift using Kolmogorov-Smirnov test.

    Args:
        train_data_path: Path to training CSV (from data_pipeline/data/train.csv)
        api_logs_path: Path to API request logs (JSONL format from logs/api_requests.jsonl)
        output_path: Where to save the drift report

    TODO:
        1. Load training data from train_data_path
        2. Load API logs from api_logs_path (JSONL format - one JSON per line)
        3. For each audio feature:
           a. Calculate KS statistic between train and production distributions
           b. Calculate p-value
           c. Flag drift if p-value < 0.05 (or similar threshold)
        4. Compile results into drift_report
        5. Save as JSON file
        6. Return drift report dictionary

    Helpful resources:
        - scipy.stats.ks_2samp() for Kolmogorov-Smirnov test
        - pandas.read_json() with orient='records' for JSONL
    """
    logger.info(f"Loading training data from {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    logger.info(f"Loading API logs from {api_logs_path}")
    api_logs = []
    try:
        with open(api_logs_path, "r") as f:
            for line in f:
                if line.strip():
                    api_logs.append(json.loads(line))
        api_df = pd.DataFrame(api_logs)
    except FileNotFoundError:
        logger.warning(f"API logs not found at {api_logs_path}. Skipping drift analysis.")
        return {"status": "no_api_logs", "message": "No API requests logged yet"}

    if len(api_df) == 0:
        logger.warning("No API logs available for drift analysis")
        return {"status": "no_api_logs", "message": "API logs are empty"}

    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Production logs shape: {api_df.shape}")

    # Get audio features (exclude 'year' and other non-feature columns)
    audio_features = [col for col in train_df.columns if col not in ["year", "timestamp"]]
    api_features = [col for col in api_df.columns if col in audio_features]

    drift_results = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "train_samples": len(train_df),
        "production_samples": len(api_df),
        "features_analyzed": len(api_features),
        "features_with_drift": 0,
        "drifted_features": [],
        "details": {}
    }

    ks_threshold = 0.05  # p-value threshold for detecting drift

    for feature in api_features:
        if feature not in api_df.columns:
            continue

        train_dist = train_df[feature].dropna()
        prod_dist = api_df[feature].dropna()

        if len(prod_dist) == 0:
            logger.warning(f"No production data for feature {feature}")
            continue

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(train_dist, prod_dist)

        is_drift = p_value < ks_threshold

        drift_results["details"][feature] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drift_detected": is_drift,
            "train_mean": float(train_dist.mean()),
            "prod_mean": float(prod_dist.mean()),
            "train_std": float(train_dist.std()),
            "prod_std": float(prod_dist.std())
        }

        if is_drift:
            drift_results["features_with_drift"] += 1
            drift_results["drifted_features"].append(feature)
            logger.warning(f"DRIFT DETECTED in {feature}: p-value={p_value:.4f}")
        else:
            logger.info(f"No drift in {feature}: p-value={p_value:.4f}")

    # Determine overall drift status
    drift_percentage = (drift_results["features_with_drift"] / drift_results["features_analyzed"] * 100) if drift_results["features_analyzed"] > 0 else 0
    drift_results["drift_percentage"] = drift_percentage
    drift_results["status"] = "DRIFT_DETECTED" if drift_percentage > 20 else "NORMAL"

    logger.info(f"\nDrift Analysis Summary:")
    logger.info(f"  Status: {drift_results['status']}")
    logger.info(f"  Features with drift: {drift_results['features_with_drift']}/{drift_results['features_analyzed']} ({drift_percentage:.1f}%)")

    # Save report
    with open(output_path, "w") as f:
        json.dump(drift_results, f, indent=2)

    logger.info(f"Drift report saved to {output_path}")

    return drift_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--api_logs", type=str, required=True)
    parser.add_argument("--output", type=str, default="drift_report.json")
    args = parser.parse_args()

    analyze_drift(args.train_data, args.api_logs, args.output)
