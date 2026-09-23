"""Model definition for ChestXpert.

The first baseline uses a pretrained ResNet-18 backbone and replaces the
classification head with a 14-output sigmoid-compatible head for
multi-label chest X-ray classification.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 14, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(model: nn.Module, path: str, epoch: int, labels: list[str]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "labels": labels,
        },
        path,
    )
