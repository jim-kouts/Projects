# Train a small CNN on STL10 and save artifacts.

#
# Outputs:
#   outputs/run_stl10/img_size={img_size}/
#       model.pt                 (LAST epoch model)
#       model_best.pt            (BEST val model)
#       metrics.json
#       config.json
#       training_curve.png
#       confusion_matrix.png

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

from model import SmallCNN


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    # Fix Python/NumPy/PyTorch randomness for more repeatable results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train SmallCNN on STL10 (img_size=64 vs 96)")

    # Data / output
    parser.add_argument("--data_dir", type=str, default="data", help="Where STL10 will be downloaded/stored")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base outputs folder")

    # Training setup
    parser.add_argument("--img_size", type=int, default=64, help="Resize images to this size (64 or 96)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (try 30 for better results)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (CPU-friendly: 128 for 64px, 64 for 96px)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 safest on many CPUs)")

    # Step 6: validation split + early stopping
    parser.add_argument("--val_frac", type=float, default=0.10, help="Fraction of train set used for validation")
    parser.add_argument("--early_stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait for val improvement before stopping")

    # Step 5: scheduler settings
    parser.add_argument("--step_size", type=int, default=10, help="StepLR step size (epochs)")
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR gamma")

    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Device (CPU in your case, but keep it general)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Output directory layout
    # -----------------------------
    run_dir = os.path.join(args.output_dir, "run_stl10", f"img_size={args.img_size}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config so you always know how a run was produced
    config = {
        "dataset": "STL10",
        "img_size": args.img_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": device,
        "val_frac": args.val_frac,
        "early_stop": args.early_stop,
        "patience": args.patience,
        "scheduler": {"type": "StepLR", "step_size": args.step_size, "gamma": args.gamma},
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=== Config ===")
    print(json.dumps(config, indent=2))
    print("Run dir:", run_dir)

    # -----------------------------
    # Transforms
    # -----------------------------
    # We'll do:
    # - Resize to slightly bigger
    # - RandomCrop to img_size
    # - Horizontal flip
    # - ToTensor
    # - Normalize
    norm = transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713)
    )

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size + 8, args.img_size + 8)),
        transforms.RandomCrop((args.img_size, args.img_size)),   # <- important
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        norm,
    ])

    test_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        norm,
    ])

    # -----------------------------
    # Load STL10 datasets
    # -----------------------------
    full_train_ds = datasets.STL10(root=args.data_dir, split="train", download=True, transform=train_tf)
    test_ds = datasets.STL10(root=args.data_dir, split="test", download=True, transform=test_tf)

    class_names = full_train_ds.classes  # list of 10 class names

    # -----------------------------
    # Create a validation split from train
    # -----------------------------
    # STL10 train split is only 5000 images, so using e.g. 10% for validation is reasonable.
    n_total = len(full_train_ds)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val

    # random_split uses PyTorch RNG; seed is set above for reproducibility
    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val])

    # Validation should NOT use random cropping/flips; it should be deterministic like test.
    # Easiest way (simple): create a separate dataset for validation with test_tf.
    # We can re-load STL10 with test_tf and then take the same indices.
    full_train_ds_eval = datasets.STL10(root=args.data_dir, split="train", download=False, transform=test_tf)
    val_ds = torch.utils.data.Subset(full_train_ds_eval, val_ds.indices)
    train_ds = torch.utils.data.Subset(full_train_ds, train_ds.indices)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Test size: {len(test_ds)}")

    # -----------------------------
    # Model, loss, optimizer, scheduler
    # -----------------------------
    model = SmallCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    # This reduces the LR after every step_size epochs (multiplies LR by gamma).
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
    )

    # -----------------------------
    # Training loop
    # -----------------------------
    train_losses = []
    val_accs = []
    test_accs = []
    lrs = []

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None

    # Early stopping tracking
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Track LR (useful to save in metrics)
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(float(current_lr))

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

        avg_train_loss = loss_sum / max(n_batches, 1)
        train_losses.append(float(avg_train_loss))

        # -----------------------------
        # Evaluate on validation + test
        # -----------------------------
        model.eval()

        # Validation accuracy
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / max(val_total, 1)
        val_accs.append(float(val_acc))

        # Test accuracy (monitoring only)
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == y).sum().item()
                test_total += y.size(0)

        test_acc = test_correct / max(test_total, 1)
        test_accs.append(float(test_acc))

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

        # Step the scheduler AFTER the epoch (standard pattern)
        scheduler.step()

        # -----------------------------
        # Save best model by validation accuracy
        # -----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            torch.save(
                {
                    "model_state_dict": best_state,
                    "img_size": args.img_size,
                    "class_names": class_names,
                    "best_epoch": best_epoch,
                    "best_val_acc": float(best_val_acc),
                },
                os.path.join(run_dir, "model_best.pt"),
            )

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Optional early stopping
        if args.early_stop and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no val improvement for {args.patience} epochs.")
            break

    # -----------------------------
    # Final evaluation: confusion matrix on TEST using BEST model
    # -----------------------------
    # Load best model weights (so confusion matrix matches the best checkpoint)
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    cm = confusion_matrix(all_true, all_preds, labels=list(range(10)))

    # -----------------------------
    # Save LAST model (end of training loop) too
    # -----------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "img_size": args.img_size,
            "class_names": class_names,
        },
        os.path.join(run_dir, "model.pt"),
    )

    # -----------------------------
    # Save metrics
    # -----------------------------
    metrics = {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(train_losses[-1]) if len(train_losses) else None,
        "final_val_acc": float(val_accs[-1]) if len(val_accs) else None,
        "final_test_acc": float(test_accs[-1]) if len(test_accs) else None,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "test_accs": test_accs,
        "lrs": lrs,
        "epochs_ran": int(len(train_losses)),
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # -----------------------------
    # Plot training curve
    # -----------------------------
    plt.figure(figsize=(10, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(train_losses, label="train loss", color="tab:blue")
    ax2.plot(val_accs, label="val acc", color="tab:orange")
    ax2.plot(test_accs, label="test acc", color="tab:green")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy")
    ax1.set_title(f"Training curve (STL10, img_size={args.img_size})")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curve.png"), dpi=200)
    plt.close()

    # -----------------------------
    # Plot confusion matrix
    # -----------------------------
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (BEST model, STL10, img_size={args.img_size})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    plt.xticks(ticks=np.arange(10), labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(10), labels=class_names)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    print("\n=== Done ===")
    print(f"Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print("Saved to:", run_dir)
    print("Files: model.pt, model_best.pt, metrics.json, config.json, training_curve.png, confusion_matrix.png")


if __name__ == "__main__":
    main()

