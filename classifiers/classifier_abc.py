from typing import ClassVar, Dict, Type
from abc import ABC, abstractmethod
import shutil
import os
import numpy as np
from torch import save, no_grad
import torch
from tqdm import tqdm
from enum import Enum


CLASSIFIER_REGISTRY: Dict[str, Type["ClassifierABC"]] = {}

__all__ = ["ClassifierABC", "CLASSIFIER_REGISTRY"]


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"


class DataLoaders:
    def __init__(self, train_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader


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
        )

        return classifier_cls(
            model_config=model_config,
            data_loaders=data_loaders,
            train_epochs=train_epochs,
        )

    @staticmethod
    def save_checkpoint(state, is_best, checkpoint):
        head, tail = os.path.split(checkpoint)
        if not os.path.exists(head):
            os.makedirs(head)

        filename = os.path.join(head, "{0}_checkpoint.pth.tar".format(tail))
        save(state, filename)
        if is_best:
            shutil.copyfile(
                filename, os.path.join(head, "{0}_best.pth.tar".format(tail))
            )

        return

    def test(self):
        if self.data_loaders.test_loader is None:
            raise ValueError(f"test loader for classifier {self} has not been set.")
        self.model_config.model.eval()
        top1 = 0
        test_loss = 0.0

        with no_grad():
            for data, target in tqdm(self.data_loaders.test_loader):
                data, target = data.to(self.model_config.device), target.to(
                    self.model_config.device
                )
                output = self.model_config.model(data)
                test_loss += self.model_config.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                top1 += pred.eq(target.view_as(pred)).sum().item()

        top1_acc = 100.0 * top1 / len(self.data_loaders.test_loader.sampler)

        return top1_acc

    @abstractmethod
    def train_step(self):
        raise NotImplementedError

    def train(self):
        if self.model_config.checkpoint is None:
            raise ValueError("Specify a valid checkpoint")
        if self.data_loaders.train_loader is None:
            raise ValueError("Train loader has not been specified for {self}.")

        best_accuracy = 0.0

        losses = []
        accuracies = []

        for epoch in range(1, self.train_epochs + 1):
            self.model_config.model.train()
            epoch_losses = self.train_step()
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = self.model_config.optimizer.param_groups[0]["lr"]
            test_accuracy = self.test()
            accuracies.append(test_accuracy)
            if self.model_config.scheduler:
                self.model_config.scheduler.step()
            is_best = test_accuracy > best_accuracy
            if is_best:
                best_accuracy = test_accuracy

            print(
                "Train Epoch {0}\t Loss: {1:.6f}\t Test Accuracy {2:.3f} \t lr: {3:.4f}".format(
                    epoch, epoch_losses.mean(), test_accuracy, lr
                )
            )
            print("Best accuracy: {:.3f} ".format(best_accuracy))

            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model_config.model.state_dict(),
                    "best_accuracy": best_accuracy,
                    "optimizer": self.model_config.optimizer.state_dict(),
                    "criterion": self.model_config.criterion,
                },
                is_best,
                self.model_config.checkpoint,
            )
