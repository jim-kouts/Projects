# A small CNN classifier for STL10 (10 classes, RGB).
# Designed to work with both img_size=64 and img_size=96.
# Key idea: use AdaptiveAvgPool2d so the classifier head doesn't depend on image size.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    A simple CNN:
      - 4 convolution blocks (Conv -> ReLU -> MaxPool)
      - Adaptive average pooling to handle different input sizes (64 or 96)
      - A small fully-connected head for classification

    Why AdaptiveAvgPool2d:
      If we used a fixed flatten size, changing input resolution would break the linear layer.
      AdaptiveAvgPool2d((1,1)) always outputs [B, C, 1, 1] no matter the input H,W.
      That makes the final linear layer stable for both 64 and 96.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Block 1:
        # Input is RGB -> 3 channels.
        # We learn 32 feature maps, then pool to reduce spatial size.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H,W

        # Block 2:
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3:
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Adaptive pooling outputs a fixed spatial size regardless of input resolution.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier: from 128 pooled features -> num_classes
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 3
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Global average pooling
        x = self.gap(x)              # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)    # [B, 256] , is used to flatten the tensor

        # Class logits (raw scores). CrossEntropyLoss expects logits.
        x = self.fc(x)               # [B, num_classes]
        return x
