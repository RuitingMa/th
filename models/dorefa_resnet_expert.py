import torch
import torch.nn as nn
import torch.nn.functional as F

from .dorefa_layers import DOREFAConv2d as Conv
from .dorefa_layers import DOREFALinear as Linear
from .model_abc import ModelABC

__all__ = ["ResNetExpert"]

def conv3x3(in_planes, out_planes, wbit, abit, stride=1):
    """3x3 convolution with padding"""
    return Conv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        nbit_w=wbit,
        nbit_a=abit,
    )

def conv1x1(in_planes, out_planes, wbit, abit, stride=1):
    """1x1 convolution"""
    return Conv(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        nbit_w=wbit,
        nbit_a=abit,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, wbit, abit, stride=1):
        super(BasicBlock, self).__init__()

        self.bb = nn.Sequential(
            conv3x3(in_planes, planes, wbit=wbit, abit=abit, stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes, wbit=wbit, abit=abit, stride=1),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    wbit=wbit,
                    abit=abit,
                    stride=stride,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.bb(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetExpert(nn.Module, ModelABC):
    """
    A specialized version of ResNet that may contain additional or altered layers 
    to cater to specific tasks or datasets, optimized using DoReFa quantization.
    """
    TYPE = "resnet_expert"

    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], wbit=1, abit=1, num_classes=10, custom_config=None):
        super(ResNetExpert, self).__init__()
        self.in_planes = 64

        self.head = nn.Sequential(
            Conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, nbit_w=wbit, nbit_a=abit),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], wbit, abit, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], wbit, abit, stride=2)
        if custom_config and custom_config.get('add_custom_layer', False):
            self.custom_layer = self._make_custom_layer(256, 256, wbit, abit)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], wbit, abit, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], wbit, abit, stride=2)

        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Linear(512 * block.expansion, num_classes, nbit_w=wbit),
        )

    def _make_layer(self, block, planes, num_blocks, wbit, abit, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, wbit, abit, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_custom_layer(self, in_planes, out_planes, wbit, abit):
        return nn.Sequential(
            Conv(in_planes, out_planes, kernel_size=1, stride=1, bias=False, nbit_w=wbit, nbit_a=abit),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return

    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if hasattr(self, 'custom_layer'):
            x = self.custom_layer(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.tail(x)
        return x
