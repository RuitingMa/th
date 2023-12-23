import torch.nn as nn
from .xnor_layers import *
from .model_abc import ModelABC

__all__ = ["LeNetExpert"]

"""Second version of LeNet that is smaller than the normal version
Main architecture used by the expert classifers in the ensemble"""


class LeNetExpert(nn.Module, ModelABC):
    TYPE = "lenet_expert"

    def __init__(self, out_classes=10):
        super(LeNetExpert, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=1),  # 3,5
            nn.BatchNorm2d(3, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            XNORConv2d(3, 3, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            BNLinearReLU(48, 4),  # 250, 50
            nn.BatchNorm1d(4, eps=1e-4, momentum=0.1, affine=False),
            nn.Linear(4, out_classes),
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
        x = self.classifier(x)
        return x


# def lenet_expert(out_classes=10):
#     return lenet_expert(out_classes)
