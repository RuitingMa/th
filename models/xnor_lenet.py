import torch
import torch.nn as nn
from .xnor_layers import *
from .model_abc import ModelABC

__all__ = ["LeNet5"]


class LeNet5(nn.Module, ModelABC):
    """Represents a LeNet5 model."""

    TYPE = "lenet5"

    def __init__(self, out_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1),
            nn.BatchNorm2d(5, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            XNORConv2d(5, 5, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            BNLinearReLU(80, 5),
            nn.BatchNorm1d(5, eps=1e-4, momentum=0.1, affine=False),
            nn.Linear(5, out_classes),
        )

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, "data"):
                    m.weight.data.zero_().add_(1.0)

    def norm_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, "data"):
                    m.weight.data.clamp_(min=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.norm_bn()
        x = self.features(x)
        x = self.classifier(x)
        return x
