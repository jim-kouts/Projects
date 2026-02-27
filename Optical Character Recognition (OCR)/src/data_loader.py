"""
data_loader.py
==============
Script 1/7 - Download and subset the FUNSD dataset

What this script does:
- Downloads the FUNSD dataset from HuggingFace
- Keeps a small subset (default 30 train, 10 test) since we have no GPU
- Saves the subset locally to data/raw/funsd_subset/

FUNSD (Form Understanding in Noisy Scanned Documents):
  A dataset of real scanned forms where every word is labeled as
  question, answer, header or other. Used to teach the model
  what role each word plays in a document.
"""

import argparse
import sys
import os

# Make sure utils/ is importable when running from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset, DatasetDict
from utils.config import RAW_DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def download_and_subset(output_dir: str, subset_size: int):
    """
    Download FUNSD from HuggingFace and keep a small subset for training.

    Args:
        output_dir:   Where to save the subset on disk
        subset_size:  How many training samples to keep
    """

    # Step 1: Download full dataset
    logger.info("Downloading FUNSD dataset from HuggingFace...")
    dataset = load_dataset("nielsr/funsd", trust_remote_code=True)
    logger.info(f"Full dataset — Train: {len(dataset['train'])}  Test: {len(dataset['test'])}")

    # Step 2: Create subset
    logger.info(f"Creating subset — keeping {subset_size} train samples...")
    small_train = dataset["train"].select(range(subset_size))
    small_test  = dataset["test"].select(range(min(10, len(dataset["test"]))))

    small_dataset = DatasetDict({
        "train": small_train,
        "test":  small_test,
    })

    logger.info(f"Subset — Train: {len(small_dataset['train'])}  Test: {len(small_dataset['test'])}")

    # Step 3: Save to disk
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving subset to: {output_dir}")
    small_dataset.save_to_disk(output_dir)

    logger.info("Dataset saved successfully.")
    logger.info(f"Column names: {small_dataset['train'].column_names}")

    # Preview first sample
    first = small_dataset["train"][0]
    logger.info(f"First sample preview:")
    logger.info(f"  words    : {first['words'][:5]} ...")
    logger.info(f"  ner_tags : {first['ner_tags'][:5]} ...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download and subset FUNSD dataset"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Where to save the dataset (default: from config.py)"
    )

    parser.add_argument(
        "--subset_size",
        type=int,
        default=30,
        help="Number of training samples to keep (default: 30)"
    )

    args = parser.parse_args()

    logger.info("=== data_loader.py started ===")
    download_and_subset(args.output_dir, args.subset_size)
    logger.info("=== data_loader.py complete ===")