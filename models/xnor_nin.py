import torch.nn as nn
from .xnor_layers import *
from .model_abc import ModelABC

__all__ = ["nin"]


class NIN(nn.Module, ModelABC):
    TYPE = "nin"

    def __init__(self, out_class=10):
        super(NIN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(50, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            BNConvReLU(50, 40, kernel_size=1, stride=1, padding=0),
            BNConvReLU(40, 20, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BNConvReLU(20, 50, kernel_size=5, stride=1, padding=2, dropout_ratio=0.5),
            BNConvReLU(50, 50, kernel_size=1, stride=1, padding=0),
            BNConvReLU(50, 50, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            BNConvReLU(50, 50, kernel_size=3, stride=1, padding=1, dropout_ratio=0.5),
            BNConvReLU(50, 50, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(50, eps=1e-4, momentum=0.1, affine=False),
            nn.Conv2d(50, out_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, "data"):
                    m.weight.data.zero_().add_(1.0)
        return

    def norm_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, "data"):
                    m.weight.data.clamp_(min=0.01)
        return

    def forward(self, x):
        self.norm_bn()
        x = self.features(x)
        return x


# def nin(out_classes=10):
#     return NIN(out_classes)
