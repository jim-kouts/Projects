# monitor_drift.py
# ------------------------------------------------------------
# Simple data drift monitoring between:
#   - reference dataset (train)
#   - current dataset (test or new incoming batch)
#
# This script computes drift metrics per feature:
# - mean/std shift
# - percentile shifts
# - PSI (Population Stability Index) using binned distributions
#
# Inputs (parquet):
#   data/processed/train.parquet   (reference)
#   data/processed/test.parquet    (current)
#
# Outputs:
#   reports/drift_summary.json
#   reports/drift_psi.png
#   reports/drift_mean_shift.png
# ------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # -----------------------------
    # 1) Args
    # -----------------------------
    parser = argparse.ArgumentParser(description="Simple drift monitoring (PSI + basic stats).")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory.")
    parser.add_argument("--reference_path", type=str, default="", help="Optional explicit reference parquet path.")
    parser.add_argument("--current_path", type=str, default="", help="Optional explicit current parquet path.")
    parser.add_argument("--target_col", type=str, default="target", help="Target column to drop if present.")
    parser.add_argument("--out_dir", type=str, default="reports", help="Where to save drift outputs.")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for PSI.")
    parser.add_argument("--psi_clip", type=float, default=1e-6, help="Small value to avoid log(0).")
    args = parser.parse_args()

    # -----------------------------
    # 2) Logging + output dir
    # -----------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("monitor_drift")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 3) Paths
    # -----------------------------
    if args.reference_path:
        ref_path = Path(args.reference_path)
    else:
        ref_path = Path(args.data_dir) / "processed" / "train.parquet"

    if args.current_path:
        cur_path = Path(args.current_path)
    else:
        cur_path = Path(args.data_dir) / "processed" / "test.parquet"

    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference file: {ref_path}")
    if not cur_path.exists():
        raise FileNotFoundError(f"Missing current file: {cur_path}")

    logger.info("Reference: %s", ref_path)
    logger.info("Current:   %s", cur_path)

    # -----------------------------
    # 4) Load data
    # -----------------------------
    ref_df = pd.read_parquet(ref_path)
    cur_df = pd.read_parquet(cur_path)

    # Drop target if present (we only monitor feature distributions here)
    if args.target_col in ref_df.columns:
        ref_df = ref_df.drop(columns=[args.target_col])
    if args.target_col in cur_df.columns:
        cur_df = cur_df.drop(columns=[args.target_col])

    # Keep only common columns (safety)
    common_cols = sorted(list(set(ref_df.columns).intersection(set(cur_df.columns))))
    ref_df = ref_df[common_cols]
    cur_df = cur_df[common_cols]

    # We focus on numeric columns for this simple drift script
    num_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("Common columns: %d | Numeric columns: %d", len(common_cols), len(num_cols))

    # -----------------------------
    # 5) PSI helper (simple, common in finance/risk)
    # -----------------------------
    # PSI formula over bins:
    #   PSI = sum_i (p_ref_i - p_cur_i) * ln(p_ref_i / p_cur_i)
    #
    # We create bins based on reference quantiles.
    def psi_for_feature(x_ref: np.ndarray, x_cur: np.ndarray, n_bins: int) -> float:
        # Drop NaNs
        x_ref = x_ref[~np.isnan(x_ref)]
        x_cur = x_cur[~np.isnan(x_cur)]

        # If feature is constant or empty, PSI isn't meaningful
        if len(x_ref) == 0 or len(x_cur) == 0:
            return float("nan")
        if np.all(x_ref == x_ref[0]) and np.all(x_cur == x_cur[0]):
            return 0.0

        # Quantile-based bin edges from reference
        # Using unique edges avoids issues with repeated values
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(x_ref, qs)
        edges = np.unique(edges)

        # If we can't form bins, return 0
        if len(edges) < 3:
            return 0.0

        # Histogram counts -> proportions
        ref_counts, _ = np.histogram(x_ref, bins=edges)
        cur_counts, _ = np.histogram(x_cur, bins=edges)

        ref_prop = ref_counts / max(1, ref_counts.sum())
        cur_prop = cur_counts / max(1, cur_counts.sum())

        # Avoid zeros in log ratio
        ref_prop = np.clip(ref_prop, args.psi_clip, 1.0)
        cur_prop = np.clip(cur_prop, args.psi_clip, 1.0)

        psi = np.sum((ref_prop - cur_prop) * np.log(ref_prop / cur_prop))
        return float(psi)

    # -----------------------------
    # 6) Compute drift stats per feature
    # -----------------------------
    drift_rows = []
    for col in num_cols:
        x_ref = ref_df[col].to_numpy(dtype=float)
        x_cur = cur_df[col].to_numpy(dtype=float)

        # Basic stats
        ref_mean = float(np.nanmean(x_ref))
        cur_mean = float(np.nanmean(x_cur))
        ref_std = float(np.nanstd(x_ref))
        cur_std = float(np.nanstd(x_cur))

        # Percentiles (robust)
        ref_p10, ref_p50, ref_p90 = [float(np.nanquantile(x_ref, q)) for q in (0.10, 0.50, 0.90)]
        cur_p10, cur_p50, cur_p90 = [float(np.nanquantile(x_cur, q)) for q in (0.10, 0.50, 0.90)]

        # PSI
        psi = psi_for_feature(x_ref, x_cur, args.n_bins)

        drift_rows.append(
            {
                "feature": col,
                "ref_mean": ref_mean,
                "cur_mean": cur_mean,
                "mean_shift": cur_mean - ref_mean,
                "ref_std": ref_std,
                "cur_std": cur_std,
                "std_shift": cur_std - ref_std,
                "ref_p10": ref_p10,
                "cur_p10": cur_p10,
                "ref_p50": ref_p50,
                "cur_p50": cur_p50,
                "ref_p90": ref_p90,
                "cur_p90": cur_p90,
                "psi": psi,
            }
        )

    drift_df = pd.DataFrame(drift_rows).sort_values("psi", ascending=False)

    # Simple PSI interpretation thresholds (common rule-of-thumb):
    # < 0.1  : no / minimal shift
    # 0.1-0.2: moderate shift
    # > 0.2  : significant shift
    # (These are heuristics, not laws.)
    n_sig = int((drift_df["psi"] > 0.2).sum())
    n_mod = int(((drift_df["psi"] >= 0.1) & (drift_df["psi"] <= 0.2)).sum())

    summary = {
        "reference_path": str(ref_path),
        "current_path": str(cur_path),
        "n_reference_rows": int(ref_df.shape[0]),
        "n_current_rows": int(cur_df.shape[0]),
        "n_features_checked": int(len(num_cols)),
        "n_bins": int(args.n_bins),
        "psi_thresholds": {"moderate": 0.1, "significant": 0.2},
        "n_moderate_psi": n_mod,
        "n_significant_psi": n_sig,
        "top_psi_features": drift_df.head(10)[["feature", "psi"]].to_dict(orient="records"),
    }

    # Save JSON summary
    summary_path = out_dir / "drift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved drift summary: %s", summary_path)

    # -----------------------------
    # 7) Plot PSI per feature (top 20)
    # -----------------------------
    top = drift_df.head(20).copy()
    top = top.iloc[::-1]  # reverse for nicer horizontal bar chart

    plt.figure(figsize=(8, max(4, int(0.35 * len(top)))))
    plt.barh(top["feature"], top["psi"])
    plt.axvline(0.1, linestyle="--", linewidth=1)
    plt.axvline(0.2, linestyle="--", linewidth=1)
    plt.xlabel("PSI")
    plt.title("Top PSI Drift Features (higher = more shift)")
    psi_path = out_dir / "drift_psi.png"
    plt.tight_layout()
    plt.savefig(psi_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", psi_path)

    # -----------------------------
    # 8) Plot mean shift per feature (top 20 by abs shift)
    # -----------------------------
    top_shift = drift_df.copy()
    top_shift["abs_mean_shift"] = top_shift["mean_shift"].abs()
    top_shift = top_shift.sort_values("abs_mean_shift", ascending=False).head(20)
    top_shift = top_shift.iloc[::-1]

    plt.figure(figsize=(8, max(4, int(0.35 * len(top_shift)))))
    plt.barh(top_shift["feature"], top_shift["mean_shift"])
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Mean shift (current - reference)")
    plt.title("Top Mean Shifts (signed)")
    mean_path = out_dir / "drift_mean_shift.png"
    plt.tight_layout()
    plt.savefig(mean_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", mean_path)

    logger.info("Done ✅")


if __name__ == "__main__":
    main()
