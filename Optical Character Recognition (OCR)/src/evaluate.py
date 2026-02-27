"""
evaluate.py
===========
Script 4/7 - Evaluate the fine-tuned LayoutLMv3 model on the test set

What this script does:
- Loads the best model checkpoint saved by train_layoutlm.py
- Runs inference on the test set
- Computes Precision, Recall and F1-score per label
- Plots a heatmap of the confusion matrix
- Plots a bar chart of F1-score per label

Key concepts:
- Precision: Of all the times the model predicted label X,
  how often was it actually X? (avoid false positives)
- Recall: Of all the actual X labels in the data,
  how many did the model find? (avoid false negatives)
- F1-score: Harmonic mean of Precision and Recall.
  Single number that balances both. Higher = better.
- Confusion Matrix: Table showing predicted vs actual labels.
  Diagonal = correct predictions. Off-diagonal = mistakes.
- We ignore label -100 (padding tokens) during evaluation.
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3ForTokenClassification
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    PROCESSED_DATA_DIR,
    BEST_MODEL_DIR,
    PLOTS_DIR,
    ID2LABEL,
    DEFAULT_BATCH_SIZE,
    NUM_LABELS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Dataset class (same as train_layoutlm.py) ──────────────────────────────────

class FUNSDDataset(Dataset):
    """Loads preprocessed .pt files from disk."""

    def __init__(self, folder_path):
        self.files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".pt")
        ])
        logger.info(f"Found {len(self.files)} samples in: {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=True)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_f1_per_label(f1_scores, label_names, output_dir):
    """
    Bar chart of F1-score for each NER label.
    Quickly shows which labels the model handles well vs poorly.
    """
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#AAAAAA"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(label_names, f1_scores, color=colors, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.title("F1-Score per Label", fontsize=14, fontweight="bold")
    plt.xlabel("Label", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0, 1.15)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "f1_per_label.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"[PLOT] Saved: {path}")


def plot_confusion_matrix(cm, label_names, output_dir):
    """
    Heatmap of the confusion matrix.
    Rows = actual labels, Columns = predicted labels.
    Bright diagonal = model is correct.
    Off-diagonal values = types of mistakes the model makes.
    """
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"[PLOT] Saved: {path}")


# ── Evaluation function ────────────────────────────────────────────────────────

def evaluate(args):

    # Step 1: Load test dataset
    logger.info("=== evaluate.py started ===")
    logger.info("STEP 1: Loading test dataset...")
    test_dataset = FUNSDDataset(os.path.join(args.processed_dir, "test"))
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Test batches: {len(test_loader)}")

    # Step 2: Load best model
    logger.info(f"STEP 2: Loading model from: {args.model_dir}")
    model  = LayoutLMv3ForTokenClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded OK. Device: {device}")

    # Step 3: Run inference on all test batches
    logger.info("STEP 3: Running inference on test set...")
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch       = {k: v.to(device) for k, v in batch.items()}
            outputs     = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Collect predictions and labels, ignoring padding tokens (label == -100)
            for pred_seq, label_seq in zip(predictions, batch["labels"]):
                for pred, label in zip(pred_seq, label_seq):
                    if label.item() != -100:
                        all_preds.append(pred.item())
                        all_labels.append(label.item())

            logger.info(f"  Batch {batch_idx + 1}/{len(test_loader)} done.")

    logger.info(f"Total tokens evaluated: {len(all_labels)}")

    # Step 4: Compute metrics
    logger.info("STEP 4: Computing Precision, Recall, F1-score...")
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(NUM_LABELS)),
        zero_division=0,
    )

    # Print per-label results
    logger.info(f"{'Label':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    logger.info("-" * 55)
    for i, name in enumerate(label_names):
        logger.info(f"{name:<15} {precision[i]:>10.3f} {recall[i]:>10.3f} "
                    f"{f1[i]:>10.3f} {support[i]:>10}")

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    logger.info(f"Weighted F1-score (overall): {weighted_f1:.4f}")
    logger.info("Full classification report:")
    logger.info("\n" + classification_report(
        all_labels, all_preds, target_names=label_names, zero_division=0
    ))

    # Step 5: Plot confusion matrix and F1 chart
    logger.info("STEP 5: Plotting confusion matrix and F1 chart...")
    os.makedirs(args.plots_dir, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_LABELS)))
    plot_confusion_matrix(cm, label_names, args.plots_dir)
    plot_f1_per_label(f1.tolist(), label_names, args.plots_dir)

    # Summary
    logger.info("=== EVALUATION COMPLETE ===")
    logger.info(f"Model evaluated : {args.model_dir}")
    logger.info(f"Test samples    : {len(test_dataset)}")
    logger.info(f"Tokens evaluated: {len(all_labels)}")
    logger.info(f"Weighted F1     : {weighted_f1:.4f}")
    logger.info(f"Plots saved to  : {args.plots_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned LayoutLMv3 on the FUNSD test set"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(BEST_MODEL_DIR),
        help="Path to saved model (default: from config.py)"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Path to processed dataset (default: from config.py)"
    )

    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Where to save evaluation plots (default: from config.py)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference (default: from config.py)"
    )

    args = parser.parse_args()
    evaluate(args)