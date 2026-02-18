# validate_data.py
# ------------------------------------------------------------
# Validate processed train/test datasets.
#
# This script acts like a "data contract":
# if validation fails, training should NOT proceed.
#
# Inputs:
#   data/processed/train.parquet
#   data/processed/test.parquet
#
# Outputs:
#   reports/validation_report.json
#
# ------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # -----------------------------
    # 1) Parse arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Validate processed train/test data.")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory.")
    parser.add_argument("--target_col", type=str, default="target", help="Target column name.")
    parser.add_argument(
        "--max_missing_frac",
        type=float,
        default=0.30,
        help="Fail if any column has missing fraction above this threshold.",
    )
    parser.add_argument(
        "--max_duplicates_frac",
        type=float,
        default=0.01,
        help="Warn/fail if duplicates exceed this fraction (simple heuristic).",
    )
    args = parser.parse_args()

    # -----------------------------
    # 2) Logging
    # -----------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("validate_data")

    # Ensure reports folder exists
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 3) Locate files
    # -----------------------------
    data_dir = Path(args.data_dir)
    train_path = data_dir / "processed" / "train.parquet"
    test_path = data_dir / "processed" / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")

    logger.info("Reading train: %s", train_path)
    train_df = pd.read_parquet(train_path)

    logger.info("Reading test:  %s", test_path)
    test_df = pd.read_parquet(test_path)

    # We'll collect results here and write them to JSON at the end
    report = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "passed": True,          # assume pass until a hard failure happens
        "errors": [],            # hard failures
        "warnings": [],          # soft issues
        "stats": {},             # useful numbers
    }

    # Helper: record an error and mark as failed
    def fail(msg: str):
        logger.error(msg)
        report["errors"].append(msg)
        report["passed"] = False

    # Helper: record a warning (doesn't fail by default)
    def warn(msg: str):
        logger.warning(msg)
        report["warnings"].append(msg)

    # -----------------------------
    # 4) Basic shape checks
    # -----------------------------
    report["stats"]["train_shape"] = list(train_df.shape)
    report["stats"]["test_shape"] = list(test_df.shape)
    logger.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)

    if train_df.shape[0] == 0:
        fail("Train dataset has 0 rows.")
    if test_df.shape[0] == 0:
        fail("Test dataset has 0 rows.")

    # -----------------------------
    # 5) Column consistency checks
    # -----------------------------
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)

    if set(train_cols) != set(test_cols):
        missing_in_test = sorted(list(set(train_cols) - set(test_cols)))
        missing_in_train = sorted(list(set(test_cols) - set(train_cols)))
        fail(f"Train/test column mismatch. Missing in test: {missing_in_test} | Missing in train: {missing_in_train}")

    # Ensure same column order (optional, but helps reproducibility)
    # We'll reorder test to match train
    test_df = test_df[train_cols]

    # -----------------------------
    # 6) Target checks
    # -----------------------------
    if args.target_col not in train_df.columns:
        fail(f"Target column '{args.target_col}' missing in train.")
    if args.target_col not in test_df.columns:
        fail(f"Target column '{args.target_col}' missing in test.")

    # Only continue target checks if target exists
    if args.target_col in train_df.columns and args.target_col in test_df.columns:
        # Check binary 0/1
        train_unique = sorted(pd.unique(train_df[args.target_col].dropna()))
        test_unique = sorted(pd.unique(test_df[args.target_col].dropna()))

        report["stats"]["train_target_unique"] = [int(x) for x in train_unique] if len(train_unique) <= 10 else "too_many"
        report["stats"]["test_target_unique"] = [int(x) for x in test_unique] if len(test_unique) <= 10 else "too_many"

        # Expect exactly two classes: 0 and 1
        if set(train_unique) != {0, 1}:
            fail(f"Train target is not binary {{0,1}}. Found: {train_unique}")
        if set(test_unique) != {0, 1}:
            fail(f"Test target is not binary {{0,1}}. Found: {test_unique}")

        # Check class balance roughly consistent
        train_rate = float(train_df[args.target_col].mean())
        test_rate = float(test_df[args.target_col].mean())
        report["stats"]["train_target_rate"] = train_rate
        report["stats"]["test_target_rate"] = test_rate
        logger.info("Target rate train=%.4f | test=%.4f", train_rate, test_rate)

        # If difference is large, warn (doesn't automatically fail)
        if abs(train_rate - test_rate) > 0.05:
            warn(f"Target rate differs by >0.05 (train={train_rate:.4f}, test={test_rate:.4f}). Stratification might be off.")

    # -----------------------------
    # 7) Missing values checks
    # -----------------------------
    # Compute missing fraction per column
    train_missing_frac = (train_df.isna().mean()).sort_values(ascending=False)
    test_missing_frac = (test_df.isna().mean()).sort_values(ascending=False)

    # Save top missing columns in report (so you can show this in portfolio)
    report["stats"]["top_missing_train"] = train_missing_frac.head(10).round(4).to_dict()
    report["stats"]["top_missing_test"] = test_missing_frac.head(10).round(4).to_dict()

    # Hard fail if any column exceeds threshold
    bad_train_cols = train_missing_frac[train_missing_frac > args.max_missing_frac]
    bad_test_cols = test_missing_frac[test_missing_frac > args.max_missing_frac]

    if len(bad_train_cols) > 0:
        fail(f"Train has columns with missing_frac > {args.max_missing_frac}: {bad_train_cols.round(4).to_dict()}")
    if len(bad_test_cols) > 0:
        fail(f"Test has columns with missing_frac > {args.max_missing_frac}: {bad_test_cols.round(4).to_dict()}")

    # -----------------------------
    # 8) Duplicate rows checks
    # -----------------------------
    # Duplicates can signal issues in ETL / data extraction
    train_dup_count = int(train_df.duplicated().sum())
    test_dup_count = int(test_df.duplicated().sum())

    report["stats"]["train_duplicates"] = train_dup_count
    report["stats"]["test_duplicates"] = test_dup_count

    train_dup_frac = train_dup_count / max(1, train_df.shape[0])
    test_dup_frac = test_dup_count / max(1, test_df.shape[0])

    if train_dup_frac > args.max_duplicates_frac:
        warn(f"Train duplicate fraction is high: {train_dup_frac:.4f} ({train_dup_count} rows).")
    if test_dup_frac > args.max_duplicates_frac:
        warn(f"Test duplicate fraction is high: {test_dup_frac:.4f} ({test_dup_count} rows).")

    # -----------------------------
    # 9) Numeric sanity checks (inf, huge values)
    # -----------------------------
    # This is a simple check: look for +/-inf in numeric columns.
    # (Parquet + pandas can sometimes carry infs.)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != args.target_col]  # exclude target
    report["stats"]["n_numeric_cols"] = len(numeric_cols)

    if len(numeric_cols) == 0:
        warn("No numeric columns detected (unexpected for most tabular ML).")

    # Count infs in train/test
    train_inf = int(np.isinf(train_df[numeric_cols]).to_numpy().sum()) if numeric_cols else 0
    test_inf = int(np.isinf(test_df[numeric_cols]).to_numpy().sum()) if numeric_cols else 0
    report["stats"]["train_inf_values"] = train_inf
    report["stats"]["test_inf_values"] = test_inf

    if train_inf > 0:
        fail(f"Train contains {train_inf} infinite values in numeric columns.")
    if test_inf > 0:
        fail(f"Test contains {test_inf} infinite values in numeric columns.")

    # Optional: warn if extremely large absolute values exist (heuristic)
    # This is NOT always wrong, but it can reveal unit/scaling bugs.
    if numeric_cols:
        train_abs_max = float(train_df[numeric_cols].abs().max().max())
        test_abs_max = float(test_df[numeric_cols].abs().max().max())
        report["stats"]["train_abs_max"] = train_abs_max
        report["stats"]["test_abs_max"] = test_abs_max

        if train_abs_max > 1e9 or test_abs_max > 1e9:
            warn(f"Very large numeric values detected (abs max train={train_abs_max:.2e}, test={test_abs_max:.2e}). Check units/scaling.")

    # -----------------------------
    # 10) Save report JSON
    # -----------------------------
    report_path = reports_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if report["passed"]:
        logger.info("Validation PASSED ✅ Report saved to: %s", report_path)
    else:
        logger.error("Validation FAILED ❌ Report saved to: %s", report_path)
        # Return non-zero exit code behavior by raising an error
        raise SystemExit(1)


if __name__ == "__main__":
    main()
