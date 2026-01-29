# Predict script for the fine-tuned classifier (SimCLR-pretrained encoder + head).
# Saves a grid of images with true/pred labels.
#
# Usage:
#   python src/predict.py --run_dir outputs/run_stl10_finetune/img_size=96 --img_size 96

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ssl_model import EncoderCNN, Classifier


def main():
    parser = argparse.ArgumentParser(description="Predict with fine-tuned STL10 classifier")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--run_dir", type=str, required=True, help="Folder containing model_best.pt")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--n_images", type=int, default=12)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(args.run_dir, "model_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Expected model_best.pt at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Match preprocessing used in training
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

    class_names = ckpt.get("class_names", test_ds.classes)
    feat_dim = ckpt.get("feat_dim", 256)

    # Build model and load weights
    encoder = EncoderCNN(feat_dim=feat_dim)
    model = Classifier(encoder=encoder, feat_dim=feat_dim, num_classes=10)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Take one batch
    x, y = next(iter(test_loader))
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    y_true = y.numpy()

    # Unnormalize images for display
    x_cpu = x.cpu()
    mean = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
    std = torch.tensor([0.2603, 0.2566, 0.2713]).view(1, 3, 1, 1)

    x_vis = torch.clamp(x_cpu * std + mean, 0.0, 1.0).numpy()

    # Plot grid
    n = min(args.n_images, len(test_ds))
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(12, 3 * rows))

    for i in range(n):
        img = np.transpose(x_vis[i], (1, 2, 0))
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"T: {class_names[y_true[i]]}\nP: {class_names[preds[i]]}")

    plt.tight_layout()

    out_path = os.path.join(args.run_dir, "pred_examples.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
