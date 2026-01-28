"""
CNN Autoencoder for image reconstruction with a dataset switch:
- FashionMNIST (grayscale, 1 channel)
- CIFAR-10 (RGB, 3 channels)

Saves:
- outputs/run_{data_set}/model.pt
- outputs/run_{data_set}/train_loss.png
- outputs/run_{data_set}/recon_grid.png
- outputs/run_{data_set}/config.json
- outputs/run_{data_set}/metrics.json


Run examples:
  python train.py --dataset fashionmnist --epochs 10 --batch_size 128
  python train.py --dataset cifar10 --epochs 20 --batch_size 128
"""

import os
import json
import time
import argparse
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import shutil

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    """Make runs more repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# A small CNN Autoencoder
# ----------------------------
class CNNAutoencoder(nn.Module):
    """
    A simple convolutional autoencoder that works for 1-channel or 3-channel images.
    Assumes input images are square and size is divisible by 4 (e.g., 32, 64, 96).

    Architecture:
      Encoder:  C -> 32 -> 64  (downsample x2 via stride=2 twice)
      Decoder:  64 -> 32 -> C  (upsample x2 via ConvTranspose2d twice)
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Encoder: downsample twice (H,W) -> (H/4, W/4)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # /2 again
            nn.ReLU(inplace=True),
        )

        # Decoder: upsample twice back to original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # x2
            nn.Sigmoid(),  # output in [0, 1], matching ToTensor() scaling
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode to reconstruct the input."""
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


# ----------------------------
# Dataset selection
# ----------------------------
def get_dataset_and_channels(name: str, data_dir: str, img_size: int):
    """
    Returns:
      train_ds, test_ds, in_channels, class_names (optional list or None)

    We keep transforms simple:
      - Resize to img_size (default 32) so one model works for both datasets
      - ToTensor() gives values in [0, 1]
    """
    name = name.lower().strip()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # unify sizes
        transforms.ToTensor(),                    # range [0,1]
    ])

    if name == "fashionmnist":
        train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tf)
        test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tf)
        in_channels = 1
        class_names = train_ds.classes
        return train_ds, test_ds, in_channels, class_names

    if name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
        test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf)
        in_channels = 3
        class_names = train_ds.classes
        return train_ds, test_ds, in_channels, class_names

    raise ValueError("Unknown dataset. Use --dataset fashionmnist or cifar10.")


# ----------------------------
# Training loop
# ----------------------------
def train_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> list:
    """
    Train the autoencoder using reconstruction loss (MSE).
    Returns a list of average training loss per epoch.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        n_batches = 0

        for x, _ in loader:
            x = x.to(device)

            # Forward reconstruction
            x_rec = model(x)

            # Reconstruction loss
            loss = criterion(x_rec, x)

            # Backprop + update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch:03d}/{epochs} - train MSE: {avg_loss:.6f}")

    return losses


@torch.no_grad()
def evaluate_recon_loss(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Compute average reconstruction MSE over a loader (e.g., test set)."""
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    loss_sum = 0.0
    n_batches = 0

    for x, _ in loader:
        x = x.to(device)
        x_rec = model(x)
        loss = criterion(x_rec, x)
        loss_sum += loss.item()
        n_batches += 1

    return loss_sum / max(n_batches, 1)


# ----------------------------
# Plotting / saving artifacts
# ----------------------------
def save_loss_curve(losses: list, out_path: str) -> None:
    """Save training loss curve."""
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title("Training Reconstruction Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def save_recon_grid(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    out_path: str,
    n: int = 8,
) -> None:
    """
    Save a grid showing originals (top row) and reconstructions (bottom row).
    Works for grayscale or RGB.

    n = number of images to show.
    """
    model.to(device)
    model.eval()

    # Grab one batch
    x, _ = next(iter(loader))
    x = x.to(device)[:n]
    x_rec = model(x)

    # Move to CPU for plotting
    x = x.cpu()
    x_rec = x_rec.cpu()

    # Build plot: 2 rows x n columns
    plt.figure(figsize=(1.8 * n, 4))

    for i in range(n):
        # Originals
        ax1 = plt.subplot(2, n, i + 1)
        img = x[i]

        # If grayscale: shape [1,H,W] -> [H,W]
        # If RGB: shape [3,H,W] -> [H,W,3]
        if img.shape[0] == 1:
            ax1.imshow(img[0], cmap="gray")
        else:
            ax1.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax1.set_title("Orig")
        ax1.axis("off")

        # Reconstructions
        ax2 = plt.subplot(2, n, n + i + 1)
        rec = x_rec[i]
        if rec.shape[0] == 1:
            ax2.imshow(rec[0], cmap="gray")
        else:
            ax2.imshow(np.transpose(rec.numpy(), (1, 2, 0)))
        ax2.set_title("Recon")
        ax2.axis("off")

    plt.suptitle("Reconstruction Examples", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------
# Config for saving
# ----------------------------
@dataclass
class Config:
    dataset: str = "fashionmnist"
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    img_size: int = 32
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 0  # CPU-friendly default


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="CNN Autoencoder Reconstruction")

    parser.add_argument("--dataset", type=str, default="fashionmnist",
                        help="fashionmnist (grayscale) or cifar10 (RGB)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Where datasets are downloaded/stored")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Where to save outputs (runs will be subfolders)")
    parser.add_argument("--img_size", type=int, default=32,
                        help="Resize images to this size (recommended: 32 for these datasets)")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers. Use 0 for most CPU setups.")

    args = parser.parse_args()

    cfg = Config(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create run folder with dataset name only 
    run_name = f"run_{cfg.dataset.lower()}"
    run_dir = os.path.join(cfg.output_dir, run_name)

    # OPTIONAL: clear old outputs for this dataset so you don't keep stale files
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg) | {"device": device}, f, indent=2)

    print(f"Device: {device}")
    print(f"Dataset: {cfg.dataset}")
    print(f"Data dir: {cfg.data_dir}")
    print(f"Run dir:  {run_dir}")

    # Load dataset
    train_ds, test_ds, in_channels, class_names = get_dataset_and_channels(
        cfg.dataset, cfg.data_dir, cfg.img_size
    )

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Build model
    model = CNNAutoencoder(in_channels=in_channels)

    # Train
    train_losses = train_autoencoder(
        model=model,
        loader=train_loader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )

    # Evaluate recon loss on test set (just a simple metric)
    test_mse = evaluate_recon_loss(model, test_loader, device)

    # Save plots
    save_loss_curve(train_losses, os.path.join(run_dir, "train_loss.png"))
    save_recon_grid(model, test_loader, device, os.path.join(run_dir, "recon_grid.png"), n=8)

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": in_channels,
            "dataset": cfg.dataset,
            "img_size": cfg.img_size,
        },
        os.path.join(run_dir, "model.pt"),
    )

    # Save metrics
    metrics = {
        "final_train_mse": float(train_losses[-1]),
        "test_mse": float(test_mse),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Done ===")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {run_dir}")
    print("Files: model.pt, train_loss.png, recon_grid.png, config.json, metrics.json")


if __name__ == "__main__":
    main()
