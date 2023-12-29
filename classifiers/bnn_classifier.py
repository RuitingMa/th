import os
from typing import Optional
import torch
import numpy as np
import torch.nn.functional as nnf
from torch import save, no_grad
from tqdm import tqdm
from models.xnor_layers import XNORConv2d
from .classifier_abc import ClassifierABC, DataLoaders, ModelConfig
import shutil


class BnnClassifier(ClassifierABC):
    NAME = "bnn"

    def __init__(
        self,
        model_config: ModelConfig,
        data_loaders: DataLoaders,
        train_epochs: Optional[int] = 100,
    ):
        super().__init__(model_config, data_loaders, train_epochs)

    def train_step(self):
        losses = []
        for data, target in tqdm(
            self.data_loaders.train_loader, total=len(self.data_loaders.train_loader)
        ):
            data, target = data.to(self.model_config.device), target.to(
                self.model_config.device
            )
            output = self.model_config.model(data)
            loss = self.model_config.criterion(output, target)
            losses.append(loss.item())
            self.model_config.optimizer.zero_grad()
            loss.backward()
            for p in self.model_config.model.modules():
                if hasattr(p, "weight_org"):
                    p.weight.data.copy_(p.weight_org)
            self.model_config.optimizer.step()
            for p in self.model_config.model.modules():
                if hasattr(p, "weight_org"):
                    p.weight_org.data.copy_(p.weight.data.clamp_(-1, 1))
        return losses
