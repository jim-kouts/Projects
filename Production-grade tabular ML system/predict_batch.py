# predict_batch.py
# ------------------------------------------------------------
# Batch inference script (parquet-only).
#
# Why this matters (portfolio):
# - Companies rarely do "one-off" predictions.
# - They often run scheduled jobs that score many rows at once
#   and write outputs back to storage (parquet/csv/db).
#
# Inputs:
#   models/model.joblib
#   <input parquet>  (default: data/processed/test.parquet)
#
# Outputs:
#   predictions/predictions.parquet
#   predictions/prediction_summary.json
# ------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def main():
    # -----------------------------
    # 1) Parse args
    # -----------------------------
    parser = argparse.ArgumentParser(description="Batch prediction on a parquet file.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/test.parquet",
        help="Path to input parquet (must contain features; may contain target too).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save prediction outputs.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="target",
        help="If present in input, we drop it before predicting.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold to convert prob -> class label.",
    )
    args = parser.parse_args()

    # -----------------------------
    # 2) Logging
    # -----------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("predict_batch")

    # -----------------------------
    # 3) Check paths + create output dir
    # -----------------------------
    input_path = Path(args.input_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 4) Load model
    # -----------------------------
    logger.info("Loading model: %s", model_path)
    clf = joblib.load(model_path)

    # -----------------------------
    # 5) Load input data
    # -----------------------------
    logger.info("Reading input parquet: %s", input_path)
    df = pd.read_parquet(input_path)

    n_rows_before = df.shape[0]
    logger.info("Input shape: %s", df.shape)

    # If the input contains the target column, drop it (we don't want leakage)
    if args.target_col in df.columns:
        logger.info("Dropping target column '%s' from input before prediction.", args.target_col)
        X = df.drop(columns=[args.target_col])
    else:
        X = df

    # -----------------------------
    # 6) Predict probabilities + labels
    # -----------------------------
    # Probability of class 1 ("default")
    prob = clf.predict_proba(X)[:, 1]

    # Convert probabilities to labels using threshold
    pred_label = (prob >= args.threshold).astype(int)

    # -----------------------------
    # 7) Build output dataframe
    # -----------------------------
    # We keep the original input columns and append predictions at the end.
    # In production you might keep only IDs + predictions, but for demo this is fine.
    out_df = df.copy()
    out_df["prob_default"] = prob
    out_df["pred_label"] = pred_label

    # -----------------------------
    # 8) Save predictions parquet
    # -----------------------------
    out_path = output_dir / "predictions.parquet"
    out_df.to_parquet(out_path, index=False)
    logger.info("Saved predictions: %s", out_path)

    # -----------------------------
    # 9) Save a small summary JSON
    # -----------------------------
    summary = {
        "input_path": str(input_path),
        "model_path": str(model_path),
        "n_rows": int(n_rows_before),
        "threshold": float(args.threshold),
        "mean_prob_default": float(np.mean(prob)),
        "predicted_positive_rate": float(np.mean(pred_label)),
        "prob_default_min": float(np.min(prob)),
        "prob_default_max": float(np.max(prob)),
        "prob_default_p50": float(np.quantile(prob, 0.50)),
        "prob_default_p90": float(np.quantile(prob, 0.90)),
        "prob_default_p99": float(np.quantile(prob, 0.99)),
    }

    summary_path = output_dir / "prediction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved summary: %s", summary_path)
    logger.info("Done ✅")


if __name__ == "__main__":
    main()
