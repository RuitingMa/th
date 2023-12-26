import os
import torch
import numpy as np
import torch.nn.functional as nnf
from torch import save, no_grad
from tqdm import tqdm
from models.xnor_layers import XNORConv2d
from .classifier_abc import ClassifierABC
import shutil


class BnnClassifier(ClassifierABC):
    NAME = "bnn"

    def __init__(self, model, train_loader=None, test_loader=None, device=None):
        super().__init__(model, train_loader, test_loader, device)

    def train_step(self, criterion, optimizer):
        losses = []
        for data, target in tqdm(self.train_loader, total=len(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            for p in self.model.modules():
                if hasattr(p, "weight_org"):
                    p.weight.data.copy_(p.weight_org)
            optimizer.step()
            for p in self.model.modules():
                if hasattr(p, "weight_org"):
                    p.weight_org.data.copy_(p.weight.data.clamp_(-1, 1))
        return losses
