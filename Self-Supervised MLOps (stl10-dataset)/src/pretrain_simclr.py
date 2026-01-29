# Self-supervised pretraining on STL10 "unlabeled" split using SimCLR-lite.
#
# What we do:
# - For each image, make TWO random augmented views (view1, view2).
# - Encode both, push them through a projection head.
# - Contrastive loss (NT-Xent):
#     views from SAME original image should be close
#     views from DIFFERENT images should be far
#
# This script produces an encoder that (usually) learns better features than random init.

import os
import json
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ssl_model import SimCLRModel


# -----------------------------
# Simple reproducibility helper
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# "Two views" dataset wrapper
# -----------------------------
class TwoCropsTransform:
    """
    Given one image, return two independently augmented versions of it.
    This is the heart of SimCLR.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        x1 = self.base_transform(img)
        x2 = self.base_transform(img)
        return x1, x2


# -----------------------------
# NT-Xent loss (SimCLR loss)
# -----------------------------
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Minimal NT-Xent loss implementation.

    Inputs:
      z1: [B, D] projected vectors for view 1
      z2: [B, D] projected vectors for view 2

    Steps:
      1) Normalize vectors so cosine similarity = dot product
      2) Build a 2B x 2B similarity matrix
      3) For each sample, its positive is the other view of same image
      4) Use CrossEntropyLoss where the target is the positive index

    This is not the most optimized version, but it's clear and works fine for learning.
    """

    # Normalize so dot product becomes cosine similarity
    z1 = nn.functional.normalize(z1, dim=1)  # [B, D], B is batch size, D is feature dim
    z2 = nn.functional.normalize(z2, dim=1)  # [B, D]

    B = z1.size(0)

    # Stack: [2B, D]
    z = torch.cat([z1, z2], dim=0)

    # Similarity matrix: [2B, 2B]
    # Each entry (i,j) is similarity between sample i and sample j
    sim = (z @ z.t()) / temperature

    # We must NOT compare a sample with itself, so mask diagonal
    # Put a very negative number on diagonal so it contributes ~0 to softmax
    diag = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -1e9)

    # Positive pairs:
    # For i in [0..B-1], positive is i+B
    # For i in [B..2B-1], positive is i-B
    targets = torch.arange(2 * B, device=sim.device)
    targets = (targets + B) % (2 * B)

    # CrossEntropyLoss expects:
    #   input: [N, C] = sim matrix row-wise
    #   target: [N] = correct column index for each row
    loss = nn.CrossEntropyLoss()(sim, targets)
    return loss


def main():
    parser = argparse.ArgumentParser(description="SimCLR-lite pretraining on STL10 unlabeled")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--img_size", type=int, default=96, help="Try 64 first for speed, then 96")
    parser.add_argument("--epochs", type=int, default=20, help="SSL usually benefits from more epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Try 128 on CPU (or 64 if slow)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)

    parser.add_argument("--feat_dim", type=int, default=256, help="Encoder output feature size")
    parser.add_argument("--proj_dim", type=int, default=128, help="Projection head output size")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output folder for SSL pretraining
    run_dir = os.path.join(args.output_dir, "run_stl10_simclr", f"img_size={args.img_size}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config for reproducibility
    cfg = vars(args) | {"device": device}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("Device:", device)
    print("Saving to:", run_dir)

    # -----------------------------
    # Augmentations for SimCLR
    # -----------------------------
    # SimCLR needs stronger augmentations than normal classification.
    # These make two views of same image look different, so the model learns invariances.
    #
    # This is a "lite" version (still simple).
    norm = transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713),
    )

    base_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        norm,
    ])

    # Wrap so dataset returns (view1, view2)
    two_view_tf = TwoCropsTransform(base_tf)

    # Unlabeled dataset
    # Labels are dummy; we ignore them.
    unlab_ds = datasets.STL10(root=args.data_dir, split="unlabeled", download=True, transform=two_view_tf)

    loader = DataLoader(
        unlab_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,  # important: SimCLR expects consistent batch size for pairing
    )

    # Model: encoder + projection head
    model = SimCLRModel(feat_dim=args.feat_dim, proj_dim=args.proj_dim).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Train
    losses = []
    model.train()

    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        n_batches = 0

        for (x1, x2), _ in loader:
            # x1, x2: two augmented views of the same images
            x1 = x1.to(device)
            x2 = x2.to(device)

            # Forward pass: get projected vectors
            z1 = model(x1)
            z2 = model(x2)

            # Contrastive loss
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            # Backprop + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        losses.append(avg_loss)

        print(f"Epoch {epoch:03d}/{args.epochs} | SimCLR loss: {avg_loss:.4f}")

    # Save checkpoint
    # We mainly care about encoder weights; but saving full model is fine too.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "img_size": args.img_size,
            "feat_dim": args.feat_dim,
            "proj_dim": args.proj_dim,
        },
        os.path.join(run_dir, "pretrain.pt"),
    )

    # Save loss plot
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("SimCLR loss")
    plt.title(f"SimCLR-lite pretraining loss (img_size={args.img_size})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pretrain_loss.png"), dpi=200)
    plt.close()

    print("Saved:", os.path.join(run_dir, "pretrain.pt"))
    print("Saved:", os.path.join(run_dir, "pretrain_loss.png"))


if __name__ == "__main__":
    main()
