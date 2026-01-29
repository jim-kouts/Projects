# Minimal CNN encoder + SimCLR projection head + classifier head.
#
# SimCLR idea:
#   - Encoder learns representations (features) from images.
#   - Projection head maps features to a space where contrastive loss is applied.
#   - After pretraining, we discard the projection head and keep the encoder.
#   - Then we attach a classifier head and fine-tune on labeled data.

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    """
    A small CNN encoder for RGB images.

    Output: a feature vector (embedding) per image: shape [B, feat_dim]

    Notes:
    - Uses AdaptiveAvgPool2d((1,1)) so it works for img_size=64 and img_size=96.
    - This is a "feature extractor", not a classifier.
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()

        # Conv blocks: Conv -> ReLU -> MaxPool
        # Pool halves H,W each time.
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global average pooling => always [B, 256, 1, 1]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Final linear layer to choose feature dimension
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = self.gap(x)                 # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 256]
        x = self.fc(x)                  # [B, feat_dim]
        return x


class SimCLRModel(nn.Module):
    """
    SimCLR model = Encoder + Projection Head

    During pretraining:
      - We compute encoder(x) -> h
      - projection_head(h) -> z
      - We apply contrastive loss on z (not on h)

    After pretraining:
      - We keep encoder weights
      - We discard projection_head
    """

    def __init__(self, feat_dim: int = 256, proj_dim: int = 128):
        super().__init__()

        self.encoder = EncoderCNN(feat_dim=feat_dim)

        # Projection head: small MLP (linear -> ReLU -> linear)
        # This is standard in SimCLR to improve representation quality.
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)          # features
        z = self.projector(h)        # projected features for contrastive loss
        return z


class Classifier(nn.Module):
    """
    Classification model used for fine-tuning:
      encoder (pretrained) + linear classifier head (10 classes)

    For simplest setup:
      - Start by training only the classifier head (encoder frozen)
      - Then optionally unfreeze encoder and fine-tune everything
    """

    def __init__(self, encoder: EncoderCNN, feat_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)      # [B, feat_dim]
        logits = self.head(h)    # [B, 10]
        return logits
