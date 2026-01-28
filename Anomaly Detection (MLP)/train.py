"""
Multi-Layer Perceptron (MLP) Autoencoder-based anomaly detection on Fashion-MNIST.

We pick one class as "normal" (default: T-shirt/top = label 0),
train an autoencoder only on that class, then detect anomalies
(all other classes) using reconstruction error.

Artifacts saved:
- model.pt
- roc_pr.png
- recon_examples.png
- metrics.json
- train_loss.png

Requirements:
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn

Run:
  python train.py --normal_class 0 --epochs 10
"""

import os
import json
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# We use sklearn to keep ROC/PR computation simple and correct.
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# ------------------------------------------------------------
# Reproducibility helpers
# ------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch so results are repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode can reduce speed but makes runs more reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# Model: a very simple MLP autoencoder
# ------------------------------------------------------------
class MLPAutoencoder(nn.Module):
    """
    A simple fully-connected autoencoder.

    - Encoder compresses 784 pixels -> latent vector
    - Decoder expands latent vector -> 784 pixels

    This is intentionally simple and easy to follow.
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()

        # Encoder: 784 -> 256 -> 64 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        # Decoder: latent_dim -> 64 -> 256 -> 784
        # Sigmoid maps output to [0,1], matching our input scaling.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        - Flatten image
        - Encode -> latent
        - Decode -> reconstruction (flattened)
        - Return reconstruction as flattened vector
        """
        x_flat = x.view(x.size(0), -1)     # [B, 1, 28, 28] -> [B, 784]
        z = self.encoder(x_flat)           # [B, 784] -> [B, latent_dim]
        x_rec = self.decoder(z)            # [B, latent_dim] -> [B, 784]
        return x_rec


# ------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------
def get_fashion_mnist(data_dir: str):
    """
    Download/load Fashion-MNIST train and test datasets.

    We transform images to tensors scaled in [0,1].
    (FashionMNIST is grayscale, shape [1,28,28].)
    """
    tf = transforms.ToTensor()

    train_ds = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=tf
    )

    test_ds = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=tf
    )
    return train_ds, test_ds


def filter_by_class(dataset, class_id: int) -> Subset:
    """
    Return a Subset of 'dataset' that only includes samples with label == class_id.

    We iterate through dataset targets and collect indices.
    """
    # In torchvision datasets, targets is usually a list or tensor of labels
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy()
    else:
        targets_np = np.array(targets)

    indices = np.where(targets_np == class_id)[0].tolist()
    return Subset(dataset, indices)


# ------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------
def train_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> list:
    """
    Train the autoencoder using MSE reconstruction loss.

    Returns:
        train_losses: list of average loss per epoch
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # reconstruction loss

    train_losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        n_batches = 0

        for x, _ in loader:
            # Move images to device
            x = x.to(device)

            # Forward: model outputs flattened reconstruction [B, 784]
            x_rec = model(x)

            # Compute loss vs flattened original
            x_flat = x.view(x.size(0), -1)
            loss = criterion(x_rec, x_flat)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            n_batches += 1

        avg_loss = epoch_loss_sum / max(n_batches, 1)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch:03d}/{epochs} - Train MSE: {avg_loss:.6f}")

    return train_losses


@torch.no_grad()
def compute_reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple:
    """
    Compute per-image reconstruction error on a dataset.

    Returns:
        errors: numpy array [N] of reconstruction MSE per sample
        labels: numpy array [N] of integer class labels
        examples: a small dict with a few samples for visualization
    """
    model.to(device)
    model.eval()

    all_errors = []
    all_labels = []

    # We'll store a few examples (original + reconstruction) for later plotting.
    # Keep them as CPU tensors.
    example_originals = []
    example_recons = []
    example_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Forward reconstruction (flattened)
        x_rec_flat = model(x)                  # [B, 784]
        x_flat = x.view(x.size(0), -1)         # [B, 784]

        # Per-sample MSE: mean over pixels for each sample
        # mse_per_sample shape: [B]
        mse_per_sample = ((x_rec_flat - x_flat) ** 2).mean(dim=1)

        all_errors.append(mse_per_sample.detach().cpu().numpy()) 
        all_labels.append(y.detach().cpu().numpy())

        # Save a few examples for visualization (first batches only)
        if len(example_originals) < 40:
            # reshape recon to image [B,1,28,28]
            x_rec_img = x_rec_flat.view(-1, 1, 28, 28)
            example_originals.append(x.detach().cpu())
            example_recons.append(x_rec_img.detach().cpu())
            example_labels.append(y.detach().cpu())

    errors = np.concatenate(all_errors, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    examples = {
        "originals": torch.cat(example_originals, dim=0)[:40],
        "recons": torch.cat(example_recons, dim=0)[:40],
        "labels": torch.cat(example_labels, dim=0)[:40],
    }
    return errors, labels, examples


# ------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------
def plot_train_loss(train_losses: list, out_path: str) -> None:
    """
    Plot training loss vs epoch.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title("Autoencoder Training Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, out_path: str) -> dict:
    """
    Plot ROC curve and Precision-Recall curve in one image.

    y_true:
      - binary labels (1 = anomaly, 0 = normal)
    scores:
      - anomaly scores (reconstruction errors); higher => more anomalous

    Returns:
      dict with AUC and Average Precision
    """
    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    # Plot side-by-side (simple, compact)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={roc_auc:.4f})")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (AP={ap:.4f})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {"roc_auc": float(roc_auc), "average_precision": float(ap)}


def plot_recon_examples(
    model: nn.Module,
    dataset,
    normal_class: int,
    device: str,
    out_path: str,
    k_each: int = 8,
) -> None:
    """
    Save a grid of reconstruction examples:
    - top row: original normal images
    - next row: reconstructed normal images
    - next row: original anomaly images
    - next row: reconstructed anomaly images

    We select:
    - k_each random normal samples
    - k_each random anomaly samples
    """
    model.to(device)
    model.eval()

    # Collect indices for normal and anomaly samples in the dataset
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy()
    else:
        targets_np = np.array(targets)

    normal_indices = np.where(targets_np == normal_class)[0]
    anomaly_indices = np.where(targets_np != normal_class)[0]

    # Pick random indices
    normal_pick = np.random.choice(normal_indices, size=k_each, replace=False)
    anomaly_pick = np.random.choice(anomaly_indices, size=k_each, replace=False)

    # Helper to reconstruct a single image
    def reconstruct(idx: int):
        x, y = dataset[idx]
        x_in = x.unsqueeze(0).to(device)  # add batch dim
        x_rec_flat = model(x_in)
        x_rec = x_rec_flat.view(1, 1, 28, 28).detach().cpu().squeeze(0)
        return x.cpu(), x_rec, int(y)

    normals = [reconstruct(int(i)) for i in normal_pick]
    anomalies = [reconstruct(int(i)) for i in anomaly_pick]

    # Build plot: 4 rows x k_each columns
    plt.figure(figsize=(1.5 * k_each, 6))

    # Row 1: normal originals
    for i, (x, xrec, y) in enumerate(normals):
        ax = plt.subplot(4, k_each, i + 1)
        ax.imshow(x.squeeze(0), cmap="gray")
        ax.set_title("Normal")
        ax.axis("off")

    # Row 2: normal reconstructions
    for i, (x, xrec, y) in enumerate(normals):
        ax = plt.subplot(4, k_each, k_each + i + 1)
        ax.imshow(xrec.squeeze(0), cmap="gray")
        ax.set_title("Recon")
        ax.axis("off")

    # Row 3: anomaly originals
    for i, (x, xrec, y) in enumerate(anomalies):
        ax = plt.subplot(4, k_each, 2 * k_each + i + 1)
        ax.imshow(x.squeeze(0), cmap="gray")
        ax.set_title(f"Anom ({y})")
        ax.axis("off")

    # Row 4: anomaly reconstructions
    for i, (x, xrec, y) in enumerate(anomalies):
        ax = plt.subplot(4, k_each, 3 * k_each + i + 1)
        ax.imshow(xrec.squeeze(0), cmap="gray")
        ax.set_title("Recon")
        ax.axis("off")

    plt.suptitle("Autoencoder Reconstructions (Normal vs Anomaly)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST Autoencoder Anomaly Detection")

    # Data / output
    parser.add_argument("--data_dir", type=str, default="./data", help="Where to download/store Fashion-MNIST")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save outputs (runs will be subfolders)")

    # Experiment settings
    parser.add_argument("--normal_class", type=int, default=0, help="Normal class label (default 0 = T-shirt/top)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Choose device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    set_seed(args.seed)

    # Create run folder
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Normal class: {args.normal_class}")

    # Save config for reproducibility
    config = vars(args)
    config["device"] = device
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load datasets
    train_ds, test_ds = get_fashion_mnist(args.data_dir)

    # Filter training set to ONLY normal class
    train_normal_ds = filter_by_class(train_ds, args.normal_class)

    # DataLoaders
    train_loader = DataLoader(
        train_normal_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Build model
    model = MLPAutoencoder(latent_dim=args.latent_dim)

    # Train
    train_losses = train_autoencoder(
        model=model,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Save training loss plot
    plot_train_loss(train_losses, os.path.join(run_dir, "train_loss.png"))

    # Save model checkpoint
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": args.latent_dim,
            "normal_class": args.normal_class,
        },
        model_path,
    )

    # Evaluate: compute reconstruction errors on the full test set
    test_errors, test_labels, _ = compute_reconstruction_errors(model, test_loader, device)

    # Build binary ground-truth for anomaly detection:
    # anomaly = 1 if label != normal_class, else 0
    y_true = (test_labels != args.normal_class).astype(np.int32)
    scores = test_errors  # higher error => more anomalous

    # Plot ROC + PR and get summary metrics
    metrics = plot_roc_pr(y_true, scores, os.path.join(run_dir, "roc_pr.png"))

    # Also store some basic stats about errors (optional but nice)
    metrics["normal_error_mean"] = float(scores[y_true == 0].mean())
    metrics["normal_error_std"] = float(scores[y_true == 0].std())
    metrics["anomaly_error_mean"] = float(scores[y_true == 1].mean())
    metrics["anomaly_error_std"] = float(scores[y_true == 1].std())

    # Save reconstruction examples (uses the test dataset directly)
    plot_recon_examples(
        model=model,
        dataset=test_ds,
        normal_class=args.normal_class,
        device=device,
        out_path=os.path.join(run_dir, "recon_examples.png"),
        k_each=8,
    )

    # Save metrics
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Done ===")
    print(f"Saved model: {model_path}")
    print(f"Saved ROC/PR plot: {os.path.join(run_dir, 'roc_pr.png')}")
    print(f"Saved recon examples: {os.path.join(run_dir, 'recon_examples.png')}")
    print(f"Saved metrics: {os.path.join(run_dir, 'metrics.json')}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
