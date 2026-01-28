# Load a trained STL10 checkpoint (preferably model_best.pt) and run predictions
# on a few test images. Saves a grid annotated with true/pred labels.


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN


def main():
    parser = argparse.ArgumentParser(description="Predict with a trained SmallCNN on STL10")

    parser.add_argument("--data_dir", type=str, default="data", help="Where STL10 is stored/downloaded")

    # You can provide either:
    # - --run_dir (preferred): directory like outputs/run_stl10/img_size=64
    # - OR --checkpoint: full path to model_best.pt / model.pt
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory that contains model_best.pt (and metrics.json).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit path to checkpoint .pt (overrides --run_dir if provided).")

    parser.add_argument("--img_size", type=int, default=64, help="Resize size used at training time")
    parser.add_argument("--n_images", type=int, default=12, help="How many test images to show")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to save the image grid. If not provided and --run_dir is set, saves into run_dir.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Choose checkpoint path
    # -----------------------------
    # Priority:
    # 1) explicit --checkpoint
    # 2) in --run_dir: model_best.pt
    # 3) in --run_dir: model.pt
    ckpt_path = None

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        if args.run_dir is None:
            raise ValueError("Provide either --checkpoint or --run_dir.")
        best_path = os.path.join(args.run_dir, "model_best.pt")
        last_path = os.path.join(args.run_dir, "model.pt")

        if os.path.exists(best_path):
            ckpt_path = best_path
        elif os.path.exists(last_path):
            ckpt_path = last_path
        else:
            raise FileNotFoundError(f"No model_best.pt or model.pt found in run_dir: {args.run_dir}")

    # Decide output path:
    # - if user passed --output_path, use it
    # - else if user passed --run_dir, save inside it
    # - else default to pred_examples.png in current directory
    if args.output_path is not None:
        out_path = args.output_path
    elif args.run_dir is not None:
        out_path = os.path.join(args.run_dir, "pred_examples.png")
    else:
        out_path = "pred_examples.png"

    print("Using device:", device)
    print("Loading checkpoint:", ckpt_path)
    print("Saving predictions grid to:", out_path)

    # -----------------------------
    # Match training preprocessing
    # -----------------------------
    
    # In train.py you used:
    #   ToTensor() + Normalize(mean,std)
    # If we skip Normalize here, model sees different input distribution at inference.
    norm = transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713),
    )

    test_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        norm,
    ])

    test_ds = datasets.STL10(root=args.data_dir, split="test", download=False, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=args.n_images, shuffle=True, num_workers=0)

    # -----------------------------
    # Load checkpoint + model
    # -----------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")

    class_names = ckpt.get("class_names", test_ds.classes)

    # Optional info (only present for model_best.pt as saved by train.py)
    if "best_epoch" in ckpt and "best_val_acc" in ckpt:
        print(f"Best epoch: {ckpt['best_epoch']}, best_val_acc: {ckpt['best_val_acc']:.4f}")

    model = SmallCNN(num_classes=10)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # -----------------------------
    # Get a random batch and predict
    # -----------------------------
    x, y = next(iter(test_loader))
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    y_true = y.numpy()

    # For plotting, we want to show images in normal RGB space.
    # But x currently is normalized. So we unnormalize for display.
    x_cpu = x.cpu()  # tensor [B,3,H,W]

    mean = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
    std = torch.tensor([0.2603, 0.2566, 0.2713]).view(1, 3, 1, 1)

    # Undo normalization: x = x*std + mean
    x_vis = x_cpu * std + mean
    x_vis = torch.clamp(x_vis, 0.0, 1.0).numpy()

    # -----------------------------
    # Plot grid
    # -----------------------------
    n = min(args.n_images, len(test_ds))
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(12, 3 * rows))

    for i in range(n):
        img = np.transpose(x_vis[i], (1, 2, 0))  # [C,H,W] -> [H,W,C]
        true_name = class_names[y_true[i]]
        pred_name = class_names[preds[i]]

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")

        # Simple title: True vs Pred
        ax.set_title(f"T: {true_name}\nP: {pred_name}")

    plt.tight_layout()

    # Ensure output folder exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
