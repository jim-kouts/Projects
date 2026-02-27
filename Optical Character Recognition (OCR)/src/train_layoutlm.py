"""
train_layoutlm.py
=================
Script 3/7 - Fine-tuning LayoutLMv3 for NER on FUNSD

What this script does:
- Loads the processed .pt files (created by preprocessing.py)
- Creates a PyTorch DataLoader for batching
- Loads the pretrained LayoutLMv3 model
- Fine-tunes it on our FUNSD subset
- Saves the best model checkpoint to models/
- Plots the training loss curve

Key concepts:
- Fine-tuning: Taking a pretrained model and training it further on our specific task.
  We don't train from scratch — we start from weights already learned on millions of documents.
- DataLoader: PyTorch utility that batches and shuffles samples during training.
- Epoch: One full pass through the entire training dataset.
- Loss: A number that measures how wrong the model predictions are.
  Lower loss = better predictions. We want this to go down over epochs.
- Optimizer: Algorithm that adjusts model weights to reduce the loss.
  We use AdamW (Adam with Weight decay) — standard for transformers.
- Learning rate: How big each weight update step is.
  Too high = unstable training. Too low = very slow learning.
"""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import LayoutLMv3ForTokenClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    PRETRAINED_MODEL,
    ID2LABEL,
    LABEL2ID,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Dataset class ──────────────────────────────────────────────────────────────

class FUNSDDataset(Dataset):
    """
    PyTorch Dataset that loads preprocessed .pt files from disk.
    Each .pt file is one document with tensors: input_ids, attention_mask,
    bbox, labels, pixel_values.
    """

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


# ── Plot function ──────────────────────────────────────────────────────────────

def plot_loss_curve(train_losses, output_dir):
    """
    Plot training loss per epoch.
    A decreasing curve means the model is learning correctly.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             marker="o", color="#4C72B0", linewidth=2, markersize=6)
    plt.title("Training Loss per Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.xticks(range(1, len(train_losses) + 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = os.path.join(output_dir, "training_loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"[PLOT] Saved: {path}")


# ── Training function ──────────────────────────────────────────────────────────

def train(args):

    # Step 1: Load datasets
    logger.info("=== train_layoutlm.py started ===")
    logger.info("STEP 1: Loading processed datasets...")
    train_dataset = FUNSDDataset(os.path.join(args.processed_dir, "train"))
    test_dataset  = FUNSDDataset(os.path.join(args.processed_dir, "test"))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    logger.info(f"Train batches : {len(train_loader)}")
    logger.info(f"Test batches  : {len(test_loader)}")

    # Step 2: Load pretrained LayoutLMv3 model
    logger.info(f"STEP 2: Loading LayoutLMv3 from: {PRETRAINED_MODEL}")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    logger.info(f"Model loaded OK. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 3: Setup device, optimizer, scheduler
    logger.info("STEP 3: Setting up optimizer and device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device        : {device}")
    model.to(device)

    optimizer    = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps  = len(train_loader) * args.epochs
    scheduler    = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                            total_iters=total_steps)

    logger.info(f"Learning rate : {args.learning_rate}")
    logger.info(f"Epochs        : {args.epochs}")
    logger.info(f"Batch size    : {args.batch_size}")
    logger.info(f"Total steps   : {total_steps}")

    # Step 4: Training loop
    logger.info("STEP 4: Starting training...")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    train_losses = []
    best_loss    = float("inf")

    for epoch in range(1, args.epochs + 1):

        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss    = outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        logger.info(f"Epoch [{epoch}/{args.epochs}]  Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_path  = os.path.join(args.model_dir, "best_model")
            model.save_pretrained(best_path)
            logger.info(f"  --> New best model saved to: {best_path}")

    # Step 5: Save final model
    logger.info("STEP 5: Saving final model...")
    final_path = os.path.join(args.model_dir, "final_model")
    model.save_pretrained(final_path)
    logger.info(f"Final model saved to: {final_path}")

    # Step 6: Plot loss curve
    logger.info("STEP 6: Plotting training loss curve...")
    plot_loss_curve(train_losses, args.plots_dir)

    # Summary
    logger.info("=== TRAINING COMPLETE ===")
    logger.info(f"Epochs trained : {args.epochs}")
    logger.info(f"Best loss      : {best_loss:.4f}")
    logger.info(f"Best model     : {os.path.join(args.model_dir, 'best_model')}")
    logger.info(f"Final model    : {os.path.join(args.model_dir, 'final_model')}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 on FUNSD dataset for NER"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Path to processed dataset (default: from config.py)"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(MODELS_DIR),
        help="Where to save model checkpoints (default: from config.py)"
    )

    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Where to save training plots (default: from config.py)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (default: from config.py)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training (default: from config.py)"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for AdamW optimizer (default: from config.py)"
    )

    args = parser.parse_args()
    train(args)