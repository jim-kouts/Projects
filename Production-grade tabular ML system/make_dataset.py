# make_dataset.py
# ------------------------------------------------------------
# Download a tabular dataset from OpenML, do minimal cleaning,
# define target, split into train/test, and save to disk as Parquet.
#
# Why this version?
# - Some OpenML datasets do NOT return y automatically (y=None).
# - So we also support defining the target column by name.
#
# Outputs:
#   data/raw/credit_default_raw.parquet
#   data/processed/train.parquet
#   data/processed/test.parquet
#   data/processed/meta.json
# ------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import openml


def main():
    # -----------------------------
    # 1) Parse command-line args
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Download dataset from OpenML, split train/test, save as Parquet."
    )
    parser.add_argument(
        "--openml_id",
        type=int,
        default=45020,
        help="OpenML dataset ID (default: 45020).",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="y",
        help=(
            "Target column name to use if OpenML does not provide y automatically. "
            "Default matches the UCI credit default dataset."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Base output directory (default: data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2).",
    )
    args = parser.parse_args()

    # -----------------------------
    # 2) Logging (better than print)
    # -----------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("make_dataset")

    # -----------------------------
    # 3) Create folders if missing
    # -----------------------------
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"

    # You asked: create these if missing
    reports_dir = Path("reports")
    src_dir = Path("src")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Folders ready: %s, %s, %s, %s", raw_dir, processed_dir, reports_dir, src_dir)

    # -----------------------------
    # 4) Reproducibility
    # -----------------------------
    np.random.seed(args.seed)

    # -----------------------------
    # 5) Download dataset from OpenML
    # -----------------------------
    logger.info("Downloading OpenML dataset id=%d ...", args.openml_id)
    dataset = openml.datasets.get_dataset(args.openml_id)

    # Get the full dataframe (X + potential target column)
    # We try to request y too, but many datasets return y=None.
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe"
    )

    logger.info("Downloaded dataset: %s", dataset.name)
    logger.info("Raw X shape: %s | y provided by OpenML: %s", X.shape, y is not None)

    # -----------------------------
    # 6) Clean column names (important!)
    # -----------------------------
    df = X.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # We also clean the target column name the same way,
    # so user can pass it in "human" format and we still find it.
    target_col_clean = (
        str(args.target_col).strip().lower().replace(" ", "_").replace("-", "_")
    )

    # -----------------------------
    # 7) Define target y
    # -----------------------------
    # Case A: OpenML actually provided y -> we use it
    # Case B: y is None -> we pull target from dataframe by column name
    if y is not None:
        y_series = pd.Series(y).copy()
        logger.info("Using target provided by OpenML.")
    else:
        logger.info("OpenML did not provide y. Trying to use target column: '%s'", target_col_clean)

        # If the column isn't found, list columns to help you debug
        if target_col_clean not in df.columns:
            logger.error("Target column '%s' not found in dataframe columns.", target_col_clean)
            logger.error("Available columns are:\n%s", ", ".join(df.columns.tolist()))
            raise RuntimeError(
                f"Target column '{target_col_clean}' not found. "
                f"Try passing --target_col with the correct name."
            )

        # Extract target and remove it from features
        y_series = df[target_col_clean].copy()
        df = df.drop(columns=[target_col_clean])

    # -----------------------------
    # 8) Convert target to binary 0/1
    # -----------------------------
    # We store final dataframe as: features + "target"
    y_numeric = pd.to_numeric(y_series, errors="coerce")

    if y_numeric.isna().any():
        # Not numeric -> map two unique labels to 0/1
        classes = sorted(y_series.dropna().unique().tolist())
        if len(classes) != 2:
            raise RuntimeError(
                f"Target has {len(classes)} classes (expected binary). Classes: {classes}"
            )
        mapping = {classes[0]: 0, classes[1]: 1}
        target = y_series.map(mapping).astype(int)
        logger.info("Target mapping used (string labels): %s", mapping)
    else:
        uniq = sorted(pd.unique(y_numeric.dropna()))
        if len(uniq) != 2:
            raise RuntimeError(f"Target numeric unique values are {uniq} (expected binary).")
        mapping = {uniq[0]: 0, uniq[1]: 1}
        target = y_numeric.map(mapping).astype(int)
        logger.info("Target mapping used (numeric labels): %s", mapping)

    # Combine features + target
    full_df = df.copy()
    full_df["target"] = target

    # -----------------------------
    # 9) Minimal sanity checks
    # -----------------------------
    n_rows, n_cols = full_df.shape
    missing_total = int(full_df.isna().sum().sum())
    target_rate = float(full_df["target"].mean())

    logger.info("Final dataset shape (features+target): %s", full_df.shape)
    logger.info("Total missing values: %d", missing_total)
    logger.info("Target rate (mean of target): %.4f", target_rate)

    # -----------------------------
    # 10) Save raw snapshot
    # -----------------------------
    raw_path = raw_dir / "credit_default_raw.parquet"
    full_df.to_parquet(raw_path, index=False)
    logger.info("Saved raw dataset to: %s", raw_path)

    # -----------------------------
    # 11) Train/test split (stratified)
    # -----------------------------
    train_df, test_df = train_test_split(
        full_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=full_df["target"],
    )

    logger.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)
    logger.info("Train target rate: %.4f", float(train_df["target"].mean()))
    logger.info("Test  target rate: %.4f", float(test_df["target"].mean()))

    # -----------------------------
    # 12) Save processed splits
    # -----------------------------
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info("Saved train split to: %s", train_path)
    logger.info("Saved test split  to: %s", test_path)

    # -----------------------------
    # 13) Save metadata JSON
    # -----------------------------
    meta = {
        "openml_id": args.openml_id,
        "dataset_name": dataset.name,
        "n_rows": int(n_rows),
        "n_cols_including_target": int(n_cols),
        "n_features": int(n_cols - 1),
        "target_name": "target",
        "target_rate": target_rate,
        "seed": args.seed,
        "test_size": args.test_size,
        "raw_path": str(raw_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "missing_total": missing_total,
        "target_col_used": str(args.target_col),
        "openml_provided_y": bool(y is not None),
    }

    meta_path = processed_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved metadata to: %s", meta_path)
    logger.info("Done ✅")


if __name__ == "__main__":
    main()
