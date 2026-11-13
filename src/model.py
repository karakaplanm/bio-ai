import torch
import torch.nn as nn
from torchvision import models
from src import config

class VHLMutationClassifier(nn.Module):
    def __init__(self, backbone_name: str = config.BACKBONE_NAME, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.target_conv_layer = self.backbone.layer4[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

def build_model(device: torch.device = config.DEVICE) -> VHLMutationClassifier:
    return VHLMutationClassifier().to(device)

