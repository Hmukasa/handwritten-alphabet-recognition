import torch
from torch import nn


class MLP(nn.Module):
    """28x28 grayscale image -> 26 letters."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 26),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Original small CNN for a simple baseline."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 26),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class CNN(nn.Module):
    """
    Current CNN: the six-convolution model used as the 95.7% baseline.

    Architecture:
      (Conv-BN-ReLU) x 2 -> Pool
      (Conv-BN-ReLU) x 2 -> Pool
      (Conv-BN-ReLU) x 2 -> Pool
      FC 1152 -> 256 -> 26
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # 28 -> 14

            # Block 2: 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # 14 -> 7

            # Block 3: 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # 7 -> 3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 26),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity/projection shortcut."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)

        return out


class SmallResNet(nn.Module):
    """
    Small residual CNN for 28x28 EMNIST Letters.
    MPS-safe: only ordinary MaxPool2d is used.
    """
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Linear(256, 26),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)


def build_model(model_type: str):
    model_type = model_type.lower()

    if model_type == "mlp":
        return MLP()
    if model_type in {"simple_cnn", "simple-cnn"}:
        return SimpleCNN()
    if model_type in {"cnn", "current_cnn", "current-cnn"}:
        return CNN()
    if model_type in {"resnet", "small_resnet", "small-resnet"}:
        return SmallResNet()

    raise ValueError(f"Unknown model type: {model_type}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
