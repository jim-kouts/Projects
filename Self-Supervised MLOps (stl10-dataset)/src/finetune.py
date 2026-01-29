# Fine-tune (supervised) on labeled STL10 train split using a SimCLR-pretrained encoder.
#
# Simple plan:
#  1) Load pretrained encoder weights from pretrain.pt
#  2) Build classifier = encoder + linear head (10 classes)
#  3) Train head first with encoder frozen (fast and stable)
#  4) Unfreeze and fine-tune whole model with lower lr
#
# This keeps things simple but usually improves accuracy vs training from scratch.

import os
import json
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix

from ssl_model import EncoderCNN, Classifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune STL10 classifier from SimCLR-pretrained encoder")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--pretrain_ckpt", type=str, required=True, help="Path to pretrain.pt from SimCLR step")

    parser.add_argument("--epochs_head", type=int, default=5, help="Train only classifier head first")
    parser.add_argument("--epochs_ft", type=int, default=20, help="Then fine-tune full model")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_ft", type=float, default=3e-4)

    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=6)

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = os.path.join(args.output_dir, "run_stl10_finetune", f"img_size={args.img_size}")
    os.makedirs(run_dir, exist_ok=True)

    cfg = vars(args) | {"device": device}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("Device:", device)
    print("Run dir:", run_dir)

    # -----------------------------
    # Transforms (classification)
    # -----------------------------
    norm = transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713),
    )

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size + 8, args.img_size + 8)),
        transforms.RandomCrop((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        norm,
    ])

    test_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        norm,
    ])

    # Labeled train/test
    full_train_ds = datasets.STL10(root=args.data_dir, split="train", download=True, transform=train_tf)
    test_ds = datasets.STL10(root=args.data_dir, split="test", download=True, transform=test_tf)
    class_names = full_train_ds.classes

    # Validation split
    n_total = len(full_train_ds)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val
    train_ds, val_indices_ds = random_split(full_train_ds, [n_train, n_val])

    # Make validation deterministic: reload train split with test_tf and use same indices
    full_train_eval = datasets.STL10(root=args.data_dir, split="train", download=False, transform=test_tf)
    val_ds = torch.utils.data.Subset(full_train_eval, val_indices_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    # -----------------------------
    # Build encoder and load pretrained weights
    # -----------------------------
    encoder = EncoderCNN(feat_dim=args.feat_dim)

    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")

    # We saved encoder_state_dict in pretrain.pt
    encoder.load_state_dict(ckpt["encoder_state_dict"])

    # Classifier = encoder + linear head
    model = Classifier(encoder=encoder, feat_dim=args.feat_dim, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    # Track curves
    train_losses = []
    val_accs = []

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    # -----------------------------
    # Phase 1: train head only (freeze encoder)
    # -----------------------------
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr_head, weight_decay=1e-4)

    for epoch in range(1, args.epochs_head + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        train_losses.append(float(avg_loss))

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                preds = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        val_accs.append(float(val_acc))

        print(f"[HEAD] Epoch {epoch:03d}/{args.epochs_head} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

    # -----------------------------
    # Phase 2: fine-tune full model (unfreeze encoder)
    # -----------------------------
    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_ft, weight_decay=1e-4)

    for epoch in range(1, args.epochs_ft + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        train_losses.append(float(avg_loss))

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                preds = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        val_accs.append(float(val_acc))

        print(f"[FT]   Epoch {epoch:03d}/{args.epochs_ft} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        # Save best by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "img_size": args.img_size,
                    "class_names": class_names,
                    "feat_dim": args.feat_dim,
                    "best_val_acc": float(best_val_acc),
                    "best_epoch": int(best_epoch),
                },
                os.path.join(run_dir, "model_best.pt"),
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.early_stop and epochs_no_improve >= args.patience:
            print(f"Early stopping: no val improvement for {args.patience} epochs.")
            break

    # Load best model for test evaluation + confusion matrix
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    # Test accuracy + confusion matrix
    all_preds = []
    all_true = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y.numpy())

            test_correct += (preds == y.numpy()).sum()
            test_total += y.size(0)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    cm = confusion_matrix(all_true, all_preds, labels=list(range(10)))
    test_acc = float(test_correct / max(test_total, 1))

    # Save metrics
    metrics = {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "train_losses": train_losses,
        "val_accs": val_accs,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training curve
    plt.figure(figsize=(10, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(train_losses, label="train loss", color="C0")
    ax2.plot(val_accs, label="val acc", color="C1")

    ax1.set_xlabel("epoch (head + ft)")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("val accuracy")
    ax1.set_title(f"Fine-tuning curve (img_size={args.img_size})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curve.png"), dpi=200)
    plt.close()

    # Save confusion matrix
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (SimCLR-pretrained, img_size={args.img_size})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    plt.xticks(ticks=np.arange(10), labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(10), labels=class_names)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    print("Done.")
    print("Best val acc:", best_val_acc)
    print("Test acc:", test_acc)
    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()
