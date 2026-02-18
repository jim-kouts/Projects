# compare_models.py
# ------------------------------------------------------------
# Compare multiple models using their saved test_metrics.json files.
#
# Inputs:
#   reports/logreg/test_metrics.json
#   reports/hgb/test_metrics.json
#
# Outputs:
#   reports/model_comparison.json
#   reports/model_comparison.png
# ------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compare models from test metrics.")
    parser.add_argument("--logreg_metrics", type=str, default="reports/logreg/test_metrics.json")
    parser.add_argument("--hgb_metrics", type=str, default="reports/hgb/test_metrics.json")
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("compare_models")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    def load_metrics(path_str):
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Missing metrics file: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    logreg = load_metrics(args.logreg_metrics)
    hgb = load_metrics(args.hgb_metrics)

    # Extract key metrics
    rows = [
        {"model": "logreg", **logreg["metrics_test"]},
        {"model": "hgb", **hgb["metrics_test"]},
    ]

    # Save comparison JSON
    comp = {"models": rows}
    comp_path = out_dir / "model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)
    logger.info("Saved: %s", comp_path)

    # Plot bar chart for ROC-AUC and PR-AUC
    models = [r["model"] for r in rows]
    roc = [r["roc_auc"] for r in rows]
    ap = [r["pr_auc"] for r in rows]

    x = list(range(len(models)))

    plt.figure()
    # Two bars per model
    # (Matplotlib default colors; no manual color settings)
    plt.bar([i - 0.15 for i in x], roc, width=0.3, label="ROC-AUC")
    plt.bar([i + 0.15 for i in x], ap, width=0.3, label="PR-AUC")
    plt.xticks(x, models)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Model Comparison on Test Set")
    plt.legend()

    plot_path = out_dir / "model_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", plot_path)

    logger.info("Done ✅")


if __name__ == "__main__":
    main()
