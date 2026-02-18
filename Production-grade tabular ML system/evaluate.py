# evaluate.py
# ------------------------------------------------------------
# Evaluate a trained model on the held-out test set + save plots.
#
# New additions:
# - Normalized confusion matrix plot
# - Threshold tuning plot (precision/recall/F1 vs threshold)
# ------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set + save plots.")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory.")
    parser.add_argument("--target_col", type=str, default="target", help="Target column name.")
    parser.add_argument("--model_path", type=str, default="models/model.joblib", help="Path to trained model artifact.")
    parser.add_argument("--out_dir", type=str, default="reports", help="Directory to save reports/plots.")
    parser.add_argument("--top_k_features", type=int, default=20, help="How many features to show in importance plot.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for confusion matrix plots.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("evaluate")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    logger.info("Loading model: %s", model_path)
    clf = joblib.load(model_path)

    # Load test data
    test_path = Path(args.data_dir) / "processed" / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    logger.info("Reading test data: %s", test_path)
    df = pd.read_parquet(test_path)

    if args.target_col not in df.columns:
        raise RuntimeError(f"Target column '{args.target_col}' not found in test data.")

    X_test = df.drop(columns=[args.target_col])
    y_test = df[args.target_col].astype(int)

    logger.info("Test dataset: X=%s, y=%s", X_test.shape, y_test.shape)
    logger.info("Test target rate: %.4f", float(y_test.mean()))

    # Predict
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= args.threshold).astype(int)

    # Metrics
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_path": str(model_path),
        "test_path": str(test_path),
        "threshold": float(args.threshold),
        "metrics_test": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        },
        "n_test": int(len(X_test)),
    }

    logger.info("TEST metrics: ROC-AUC=%.4f | PR-AUC=%.4f | Acc=%.4f | F1=%.4f", roc_auc, pr_auc, acc, f1)

    # Save metrics JSON
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics: %s", metrics_path)

    # -----------------------------
    # ROC curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    roc_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # -----------------------------
    # Precision-Recall curve
    # -----------------------------
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={pr_auc:.3f})")
    pr_path = out_dir / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # -----------------------------
    # Confusion matrix (counts)
    # -----------------------------
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (threshold={args.threshold})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    # annotate with counts + TN/FP/FN/TP labels
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{labels[i, j]}\n{v}", ha="center", va="center")

    cm_path = out_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # -----------------------------
    # Confusion matrix (normalized)
    # Normalize by true label row sums -> shows percentages per true class
    # -----------------------------
    cm_norm = cm.astype(float) / np.maximum(1.0, cm.sum(axis=1, keepdims=True))

    plt.figure()
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title(f"Confusion Matrix Normalized (threshold={args.threshold})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for (i, j), v in np.ndenumerate(cm_norm):
        plt.text(j, i, f"{v:.2f}", ha="center", va="center")

    cmn_path = out_dir / "confusion_matrix_normalized.png"
    plt.tight_layout()
    plt.savefig(cmn_path, dpi=150)
    plt.close()

    # -----------------------------
    # Calibration curve
    # -----------------------------
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (10 bins)")
    cal_path = out_dir / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(cal_path, dpi=150)
    plt.close()

    # -----------------------------
    # Threshold tuning plot
    # -----------------------------
    # Sweep thresholds and compute precision/recall/F1
    thresholds = np.linspace(0.05, 0.95, 19)
    precs, recs, f1s = [], [], []

    for th in thresholds:
        yp = (y_prob >= th).astype(int)
        p = precision_score(y_test, yp, zero_division=0)
        r = recall_score(y_test, yp, zero_division=0)
        f = f1_score(y_test, yp, zero_division=0)
        precs.append(p)
        recs.append(r)
        f1s.append(f)

    plt.figure()
    plt.plot(thresholds, precs, linewidth=2, label="Precision")
    plt.plot(thresholds, recs, linewidth=2, label="Recall")
    plt.plot(thresholds, f1s, linewidth=2, label="F1")
    plt.axvline(args.threshold, linestyle="--", linewidth=1, label="Current threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning (test set)")
    plt.legend()
    tt_path = out_dir / "threshold_tuning.png"
    plt.tight_layout()
    plt.savefig(tt_path, dpi=150)
    plt.close()

    # -----------------------------
    # Feature importance (LogReg coefficients)
    # -----------------------------
    try:
        pre = clf.named_steps["preprocess"]
        model = clf.named_steps["model"]
        feature_names = pre.get_feature_names_out()
        coefs = model.coef_.ravel()

        k = min(args.top_k_features, len(coefs))
        idx = np.argsort(np.abs(coefs))[::-1][:k]

        top_names = feature_names[idx]
        top_vals = coefs[idx]

        plt.figure(figsize=(8, max(4, int(0.25 * k))))
        y_pos = np.arange(len(top_names))
        plt.barh(y_pos, top_vals)
        plt.yticks(y_pos, top_names)
        plt.gca().invert_yaxis()
        plt.xlabel("Coefficient (signed)")
        plt.title(f"Top-{k} Feature Importance (LogReg Coefficients)")
        fi_path = out_dir / "feature_importance.png"
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not create feature importance plot: %s", str(e))

    logger.info("Saved plots to: %s", out_dir)
    logger.info("Done ✅")


if __name__ == "__main__":
    main()
