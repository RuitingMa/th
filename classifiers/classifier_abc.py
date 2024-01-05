from typing import ClassVar, Dict, Type
from abc import ABC, abstractmethod
import shutil
import os
import numpy as np
from torch import save, no_grad
import torch
from tqdm import tqdm
from enum import Enum
import torch.nn.functional as F


CLASSIFIER_REGISTRY: Dict[str, Type["ClassifierABC"]] = {}

__all__ = ["ClassifierABC", "CLASSIFIER_REGISTRY"]


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"


class DataLoaders:
    def __init__(self, train_loader=None, test_loader=None, labels=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.labels = labels


class ModelConfig:
    def __init__(
        self, model, optimizer, learning_rate, steps, gamma, checkpoint, device
    ):
        self.model = model
        self.optimizer = self.create_optimizer(optimizer, model, learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, steps, gamma=gamma
        )
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion.to(self.device)
        self.checkpoint = checkpoint

    def create_optimizer(self, optimizer: OptimizerType, model, learning_rate):
        if optimizer == OptimizerType.ADAM:
            return torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=1e-5
            )
        else:
            return torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5.0e-4
            )


class ClassifierABC(ABC):
    NAME: ClassVar[str]

    def __init__(
        self, model_config: ModelConfig, data_loaders: DataLoaders, train_epochs=100
    ):
        self.model_config = model_config
        self.data_loaders = data_loaders
        self.train_epochs = train_epochs

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "NAME"):
            CLASSIFIER_REGISTRY[cls.NAME] = cls

    @classmethod
    def from_config(
        cls,
        name,
        model,
        optimizer,
        learning_rate,
        steps,
        gamma,
        checkpoint,
        labels=None,
        train_epochs=100,
        train_loader=None,
        test_loader=None,
        device=None,
    ):
        try:
            classifier_cls = CLASSIFIER_REGISTRY[name]
        except KeyError:
            raise ValueError(f"{name} is not the name of a valid classifier type.")
        model_config = ModelConfig(
            model=model,
            optimizer=OptimizerType(optimizer),
            learning_rate=learning_rate,
            steps=steps,
            gamma=gamma,
            checkpoint=checkpoint,
            device=device,
        )

        data_loaders = DataLoaders(
            train_loader=train_loader,
            test_loader=test_loader,
            labels=labels,
        )

        return classifier_cls(
            model_config=model_config,
            data_loaders=data_loaders,
            train_epochs=train_epochs,
        )

    @staticmethod
    def save_checkpoint(state, checkpoint):
        head, tail = os.path.split(checkpoint)
        if not os.path.exists(head):
            os.makedirs(head)

        filename = os.path.join(head, "{0}_checkpoint.pth.tar".format(tail))
        save(state, filename)

        return

    def get_targets(self):
        targets = torch.tensor([], dtype=torch.int).to(self.model_config.device)
        with no_grad():
            for _, target in tqdm(self.data_loaders.test_loader):
                target = target.to(self.model_config.device)
                targets = torch.cat((targets, target), dim=0)
        return targets

    def test(self):
        if self.data_loaders.test_loader is None:
            raise ValueError(f"test loader for classifier {self} has not been set.")
        self.model_config.model.eval()
        top1 = 0
        test_loss = 0.0
        predictions = torch.tensor([], dtype=torch.int).to(self.model_config.device)
        confidence_scores = torch.tensor([], dtype=torch.float).to(
            self.model_config.device
        )

        with no_grad():
            for data, target in tqdm(self.data_loaders.test_loader):
                data, target = data.to(self.model_config.device), target.to(
                    self.model_config.device
                )
                output = self.model_config.model(data)
                probabilities = F.softmax(output, dim=1)
                test_loss += self.model_config.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                top1 += pred.eq(target.view_as(pred)).sum().item()
                predictions = torch.cat((predictions, pred), dim=0)
                confidence_scores = torch.cat((confidence_scores, probabilities), dim=0)

        top1_acc = 100.0 * top1 / len(self.data_loaders.test_loader.sampler)
        return top1_acc, predictions, confidence_scores

    @abstractmethod
    def train_step(self):
        raise NotImplementedError

    def train(self):
        if self.model_config.checkpoint is None:
            raise ValueError("Specify a valid checkpoint")
        if self.data_loaders.train_loader is None:
            raise ValueError("Train loader has not been specified for {self}.")

        losses = []

        for epoch in range(1, self.train_epochs + 1):
            self.model_config.model.train()
            epoch_losses = self.train_step()
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = self.model_config.optimizer.param_groups[0]["lr"]
            if self.model_config.scheduler:
                self.model_config.scheduler.step()

            print(
                "Train Epoch {0}\t Loss: {1:.6f} \t lr: {2:.4f}".format(
                    epoch, epoch_losses.mean(), lr
                )
            )

            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model_config.model.state_dict(),
                    "optimizer": self.model_config.optimizer.state_dict(),
                    "criterion": self.model_config.criterion,
                },
                self.model_config.checkpoint,
            )
