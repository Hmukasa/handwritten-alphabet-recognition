import torch
from torch import nn


class ResidualBlock(nn.Module):
    """3x3 convolution x2 + identity/projection shortcut."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)

        return out


class ResNet(nn.Module):
    """
    ResNet for EMNIST Letters.

    Input:
        (N, 1, 28, 28)

    Output:
        (N, 26) logits
    """

    def __init__(self, num_classes=26):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),          # 28 -> 14

            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2),          # 14 -> 7

            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),          # 7 -> 3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)


def parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
