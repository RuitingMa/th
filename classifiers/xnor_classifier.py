import os
import torch
import numpy as np
import torch.nn.functional as nnf
from torch import save, no_grad
from tqdm import tqdm
from models.xnor_layers import XNORConv2d
import shutil
from .classifier_abc import ClassifierABC


class XnorClassifier(ClassifierABC):
    NAME = "xnor"

    def __init__(self, model, train_loader=None, test_loader=None, device=None):
        super().__init__(model, train_loader, test_loader, device)

    def train_step(self, criterion, optimizer):
        losses = []
        self.model.train()

        for data, target in tqdm(self.train_loader, total=len(self.train_loader)):
            # print (data, target)
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            output = self.model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            loss.backward()

            for m in self.model.modules():
                if isinstance(m, XNORConv2d):
                    m.update_gradient()

            optimizer.step()

        return losses
