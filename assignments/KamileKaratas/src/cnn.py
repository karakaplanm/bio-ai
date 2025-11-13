import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBloodCellCNN(nn.Module):
    """
    A simple CNN for blood smear image classification.
    Input: 3-channel RGB image, 224x224
    Output: 3 classes -> [lymphocyte, neutrophil, monocyte]
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 224 -> 112 -> 56
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (N, 16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))   # (N, 32, 56, 56)
        x = torch.flatten(x, 1)                # (N, 32*56*56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
