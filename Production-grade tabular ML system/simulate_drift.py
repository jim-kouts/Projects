# simulate_drift.py
# ------------------------------------------------------------
# Create a "drifted" version of a dataset to demonstrate drift monitoring.
#
# Why:
# - Your real train/test likely come from the same distribution -> low drift.
# - It's useful to show the drift detector actually catches drift.
#
# Inputs:
#   data/processed/test.parquet   (default)
#
# Outputs:
#   data/processed/current_drifted.parquet
# ------------------------------------------------------------

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Simulate drift by modifying feature distributions.")
    parser.add_argument("--input_path", type=str, default="data/processed/test.parquet", help="Input parquet.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/current_drifted.parquet",
        help="Output parquet (drifted).",
    )
    parser.add_argument("--target_col", type=str, default="target", help="Target column (kept unchanged).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("simulate_drift")

    np.random.seed(args.seed)

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    logger.info("Reading: %s", in_path)
    df = pd.read_parquet(in_path)

    # Work on a copy
    drifted = df.copy()

    # Choose some numeric columns to drift.
    # You can tweak these to make PSI clearly pass 0.1 / 0.2 thresholds.
    # (We keep it deterministic and simple.)
    cols = [c for c in drifted.columns if c != args.target_col]

    # If your dataset columns are like x1..x20, these should exist:
    # We'll drift a few of them. If some don't exist, we just skip.
    drift_plan = [
        # (column_name, type_of_drift, parameter)
        ("x1", "add", 500.0),       # shift mean upward
        ("x14", "mul", 1.25),       # scale values up
        ("x15", "mul", 0.80),       # scale values down
        ("x17", "noise", 0.50),     # add gaussian noise proportional to std
    ]

    for col, mode, param in drift_plan:
        if col not in drifted.columns:
            logger.warning("Column '%s' not found, skipping drift.", col)
            continue

        x = pd.to_numeric(drifted[col], errors="coerce").astype(float)

        if mode == "add":
            # Add a constant shift
            drifted[col] = x + float(param)
            logger.info("Applied drift: %s = %s + %.3f", col, col, float(param))

        elif mode == "mul":
            # Multiply by a factor
            drifted[col] = x * float(param)
            logger.info("Applied drift: %s = %s * %.3f", col, col, float(param))

        elif mode == "noise":
            # Add gaussian noise based on std
            std = float(np.nanstd(x))
            noise = np.random.normal(loc=0.0, scale=float(param) * std, size=len(x))
            drifted[col] = x + noise
            logger.info("Applied drift: %s = %s + N(0, (%.3f*std)^2)", col, col, float(param))

        else:
            logger.warning("Unknown drift mode '%s' for col '%s' (skipping).", mode, col)

    # Save drifted dataset
    drifted.to_parquet(out_path, index=False)
    logger.info("Saved drifted dataset: %s", out_path)
    logger.info("Done ✅")


if __name__ == "__main__":
    main()
