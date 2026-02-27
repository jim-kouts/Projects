"""
preprocessing.py
================
Script 2/7 - Preprocessing Pipeline for FUNSD Dataset

What this script does:
- Loads the saved FUNSD subset (created by data_loader.py)
- Tokenizes each document using LayoutLMv3Processor
- Aligns NER labels with tokens
- Saves the processed dataset to data/processed/

Key concepts:
- LayoutLMv3 (Layout Language Model v3): A multimodal transformer that understands
  text + layout (bounding boxes) + image. State-of-the-art for document understanding.
- NER (Named Entity Recognition): Identifying entities in text (e.g. question, answer, header)
- Tokenization: Converting words into numerical IDs that the model understands
- Token-label alignment: A word can be split into multiple tokens, so we must
  assign the correct label to each token.
  Rule: first token gets the label, the rest get -100 (ignored during training loss)
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import LayoutLMv3Processor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PLOTS_DIR,
    PRETRAINED_MODEL,
    MAX_TOKEN_LENGTH,
    ID2LABEL,
    LABEL2ID,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Helper functions ───────────────────────────────────────────────────────────

def normalize_bbox(bbox, width=1000, height=1000):
    """
    Normalize bounding box to [0, 1000] range.
    LayoutLMv3 expects coordinates between 0 and 1000.
    bbox format: [x_min, y_min, x_max, y_max]
    """
    return [
        max(0, min(1000, int(1000 * bbox[0] / width))),
        max(0, min(1000, int(1000 * bbox[1] / height))),
        max(0, min(1000, int(1000 * bbox[2] / width))),
        max(0, min(1000, int(1000 * bbox[3] / height))),
    ]


def preprocess_sample(sample, processor, max_length):
    """
    Tokenize one FUNSD sample and align labels with tokens.
    Returns a dict of tensors ready for model input.
    """
    words    = sample["words"]
    boxes    = [normalize_bbox(b) for b in sample["bboxes"]]
    ner_tags = sample["ner_tags"]
    image    = sample["image"]

    if not words:
        return None

    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        word_labels=ner_tags,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Remove batch dimension: [1, N] -> [N]
    return {k: v.squeeze(0) for k, v in encoding.items()}


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_label_distribution(dataset_split, split_name, output_dir):
    """
    Bar chart showing how many times each label appears in the split.
    Helps us understand class imbalance before training.
    """
    counts = {label: 0 for label in ID2LABEL.values()}

    for sample in dataset_split:
        for tag_id in sample["ner_tags"]:
            label_name = ID2LABEL.get(tag_id, "O")
            counts[label_name] += 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#AAAAAA"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=10)

    plt.title(f"Label Distribution — {split_name} split", fontsize=14, fontweight="bold")
    plt.xlabel("Label (BIO format)", fontsize=12)
    plt.ylabel("Word count", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, f"label_distribution_{split_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"[PLOT] Saved: {path}")


def plot_token_lengths(processed_samples, split_name, output_dir):
    """
    Histogram of real token counts per document (excluding padding).
    Helps verify that max_length is appropriate for the dataset.
    """
    lengths  = [int(s["attention_mask"].sum()) for s in processed_samples]
    mean_len = np.mean(lengths)

    plt.figure(figsize=(9, 5))
    plt.hist(lengths, bins=20, color="#4C72B0", edgecolor="black", alpha=0.85)
    plt.axvline(mean_len, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {mean_len:.1f}")
    plt.title(f"Token Length Distribution — {split_name} split", fontsize=14, fontweight="bold")
    plt.xlabel("Number of real tokens (non-padding)", fontsize=12)
    plt.ylabel("Number of documents", fontsize=12)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, f"token_lengths_{split_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"[PLOT] Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):

    # Step 1: Load raw dataset
    logger.info("=== preprocessing.py started ===")
    logger.info(f"STEP 1: Loading raw FUNSD subset from: {args.raw_dir}")
    dataset = load_from_disk(args.raw_dir)
    logger.info(f"Train samples : {len(dataset['train'])}")
    logger.info(f"Test samples  : {len(dataset['test'])}")
    logger.info(f"Column names  : {dataset['train'].column_names}")

    first = dataset["train"][0]
    logger.info("First sample preview:")
    logger.info(f"  words    : {first['words'][:5]} ...")
    logger.info(f"  bboxes   : {first['bboxes'][:3]} ...")
    logger.info(f"  ner_tags : {first['ner_tags'][:5]} ...")
    logger.info(f"  image    : {type(first['image'])} size={first['image'].size}")

    # Step 2: Plot label distributions
    logger.info("STEP 2: Plotting label distributions...")
    os.makedirs(args.plots_dir, exist_ok=True)
    plot_label_distribution(dataset["train"], "train", args.plots_dir)
    plot_label_distribution(dataset["test"],  "test",  args.plots_dir)

    # Step 3: Load LayoutLMv3 processor
    logger.info(f"STEP 3: Loading LayoutLMv3 processor from: {PRETRAINED_MODEL}")
    processor = LayoutLMv3Processor.from_pretrained(
        PRETRAINED_MODEL,
        apply_ocr=False
    )
    logger.info("Processor loaded OK.")

    # Step 4: Tokenize and align labels for each split
    logger.info(f"STEP 4: Tokenizing splits (max_length={args.max_length})...")
    processed = {}

    for split_name in ["train", "test"]:
        logger.info(f"Processing '{split_name}' split...")
        samples = []

        for i, sample in enumerate(dataset[split_name]):
            try:
                result = preprocess_sample(sample, processor, args.max_length)
                if result is not None:
                    samples.append(result)
            except Exception as e:
                logger.warning(f"Skipping sample {i}: {e}")

            if (i + 1) % 10 == 0 or (i + 1) == len(dataset[split_name]):
                logger.info(f"  Progress: {i+1}/{len(dataset[split_name])}")

        processed[split_name] = samples
        logger.info(f"'{split_name}' done. {len(samples)} samples processed.")

    # Step 5: Plot token length distributions
    logger.info("STEP 5: Plotting token length distributions...")
    for split_name, samples in processed.items():
        if samples:
            plot_token_lengths(samples, split_name, args.plots_dir)
        else:
            logger.warning(f"No samples in '{split_name}' — skipping plot.")

    # Step 6: Save processed samples as .pt files
    logger.info("STEP 6: Saving processed dataset...")
    os.makedirs(args.processed_dir, exist_ok=True)

    for split_name, samples in processed.items():
        split_dir = os.path.join(args.processed_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, sample in enumerate(samples):
            torch.save(sample, os.path.join(split_dir, f"sample_{i:04d}.pt"))

        logger.info(f"Saved {len(samples)} files to: {split_dir}/")

    # Summary
    logger.info("=== PREPROCESSING COMPLETE ===")
    logger.info(f"Processed data : {args.processed_dir}")
    logger.info(f"Plots          : {args.plots_dir}")
    logger.info(f"Train samples  : {len(processed['train'])}")
    logger.info(f"Test samples   : {len(processed['test'])}")

    if processed["train"]:
        logger.info("Tensor keys and shapes (first train sample):")
        for key, tensor in processed["train"][0].items():
            logger.info(f"  {key}: {tensor.shape}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess FUNSD dataset for LayoutLMv3 training"
    )

    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Path to raw FUNSD subset (default: from config.py)"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Where to save the tokenized dataset (default: from config.py)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_TOKEN_LENGTH,
        help="Maximum token sequence length (default: from config.py)"
    )

    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Where to save plots (default: from config.py)"
    )

    args = parser.parse_args()
    main(args)