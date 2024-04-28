import shutil
from typing import ClassVar, Dict, List, Type, Tuple
from abc import ABC, abstractmethod
import os
from matplotlib import pyplot as plt
import numpy as np
from torch import save, no_grad
import torch
from tqdm import tqdm
from enum import Enum
import torch.nn.functional as F

from models.model_abc import ModelABC


CLASSIFIER_REGISTRY: Dict[str, Type["ClassifierABC"]] = {}

__all__ = ["ClassifierABC", "CLASSIFIER_REGISTRY"]


def draw_curve(values, file_name, epochs, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(
        [i for i in range(epochs)],
        values,
        marker="o",
        linestyle="-",
    )
    plt.title(f"{ylabel} per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(f"classifiers/graphs/{file_name}.png")
    plt.close()


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"


class DataLoaders:
    """
    Contains configuration for the data loaders used inside the classifier.
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader = None,
        test_loader: List[torch.utils.data.DataLoader] = None,
        labels: List[int] = None,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.labels = labels


class ModelConfig:
    """
    Contains configuration for the model used inside the classifier.
    """

    def __init__(
        self,
        model: ModelABC,
        optimizer: OptimizerType,
        learning_rate: float,
        steps: List[int],
        gamma: float,
        checkpoint: str,
        device: str,
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

    def create_optimizer(
        self, optimizer: OptimizerType, model: ModelABC, learning_rate: float
    ):
        """
        Creates an optimizer for the model using the optimizer type and learning rate.

        Args:
            optimizer: Optimizer type to be used.
            model: Model to be trained.
            learning_rate: Learning rate for the optimizer.

        Returns:
            An optimizer for the model.
        """
        if optimizer == OptimizerType.ADAM:
            return torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=1e-5
            )
        else:
            return torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5.0e-4
            )


class ClassifierABC(ABC):
    """
    Base class for all classifiers. Takes in details about the parameters required
    to test and train the classifier, including the model, data loaders, training
    epochs.
    """

    NAME: ClassVar[str]
    """
    Name of the classification method used to train the low-bitwidth model.
    This is also used to register the classifier in the config.
    """

    def __init__(
        self,
        id: int,
        model_config: ModelConfig,
        data_loaders: DataLoaders,
        train_epochs: int = 100,
    ):
        self.id = id
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
        id,
        labels=None,
        train_epochs=100,
        train_loader=None,
        test_loader=None,
        device=None,
    ) -> "ClassifierABC":
        """
        Registers a classifier from config using the provided parameters.
        Raises a ValueError if the name is not a valid classifier type.

        Args:
            name: Name of the classifier.
            model: Model to be used for classification.
            optimizer: Optimizer to be used for training.
            learning_rate: Learning rate for the optimizer.
            steps: Steps for the scheduler.
            gamma: Gamma value for the scheduler.
            checkpoint: Checkpoint to save the model.
            labels: Labels to train the model on.
            train_epochs: Number of epochs to train the model for.
            train_loader: Train loader for the model.
            test_loader: Test loader for the model.
            device: Device to run the model on.

        Returns:
            A classifier object.
        """
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
            id=id,
            model_config=model_config,
            data_loaders=data_loaders,
            train_epochs=train_epochs,
        )

    @staticmethod
    def save_checkpoint(state: Dict, checkpoint: str):
        """
        Saves the model at a given state on the disk using the checkpoint.

        Args:
            state: State of the model to be saved.
            checkpoint: Checkpoint to save the model at.
        """
        head, tail = os.path.split(checkpoint)
        if not os.path.exists(head):
            os.makedirs(head)

        filename = os.path.join(head, "{0}_checkpoint.pth.tar".format(tail))
        save(state, filename)

    def get_targets(self) -> torch.tensor:
        """
        Returns the targets (ie true labels) for the test data loader.

        Returns:
            A tensor of targets.
        """
        targets = torch.tensor([], dtype=torch.int).to(self.model_config.device)
        with no_grad():
            for _, target in tqdm(self.data_loaders.test_loader):
                target = target.to(self.model_config.device)
                targets = torch.cat((targets, target), dim=0)
        return targets

    def test(
        self, dataset: torch.utils.data.DataLoader = None
    ) -> Tuple[float, torch.tensor, torch.tensor]:
        """
        Tests the classifier on the test data loader and returns the accuracy,
        predictions and confidence scores. For this method to work, the test data
        loader must be set.

        Returns:
            Accuracy, predicted labels and confidence scores.
        """
        if self.data_loaders.test_loader is None and dataset is None:
            raise ValueError(f"test loader for classifier {self} has not been set.")
        if dataset is None:
            dataset = self.data_loaders.test_loader
        self.model_config.model.eval()
        top1 = 0
        test_losses = []
        predictions = torch.tensor([], dtype=torch.int).to(self.model_config.device)
        confidence_scores = torch.tensor([], dtype=torch.float).to(
            self.model_config.device
        )

        with no_grad():
            for data, target in tqdm(dataset):
                data, target = data.to(self.model_config.device), target.to(
                    self.model_config.device
                )
                output = self.model_config.model(data)
                probabilities = F.softmax(output, dim=1)
                test_losses.append(self.model_config.criterion(output, target).item())
                pred = output.argmax(dim=1, keepdim=True)
                top1 += pred.eq(target.view_as(pred)).sum().item()
                predictions = torch.cat((predictions, pred), dim=0)
                confidence_scores = torch.cat((confidence_scores, probabilities), dim=0)

        top1_acc = 100.0 * top1 / len(dataset.sampler)
        return top1_acc, predictions, confidence_scores, test_losses

    @abstractmethod
    def train_step(self) -> List[float]:
        """
        Implements a single training step for the classifier.
        """
        raise NotImplementedError

    def train(self):
        """
        Trains the classifier for the specified number of epochs and saves it
        on disk using the checkpoint. For this method to work, the train data loader
        and model checkpoint must be set.
        """
        if self.model_config.checkpoint is None:
            raise ValueError("Specify a valid checkpoint")
        if self.data_loaders.train_loader is None:
            raise ValueError("Train loader has not been specified for {self}.")

        best_accuracy = 0.0

        losses = []
        validation_accuracies = []
        validation_losses = []
        running_losses = []
        running_top1_accuracies = []

        for epoch in range(1, self.train_epochs + 1):
            self.model_config.model.train()
            epoch_losses, epoch_top1_accuracy = self.train_step()
            running_top1_accuracies.append(epoch_top1_accuracy)
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            running_losses.append(epoch_losses.mean())
            lr = self.model_config.optimizer.param_groups[0]["lr"]
            # validation
            test_accuracy, _, _, test_losses = self.test()
            validation_accuracies.append(test_accuracy)
            test_losses = np.array(test_losses)
            validation_losses.append(test_losses.mean())
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
            print()

            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model_config.model.state_dict(),
                    "optimizer": self.model_config.optimizer.state_dict(),
                    "criterion": self.model_config.criterion,
                },
                self.model_config.checkpoint,
            )
        if os.path.exists("classifiers/graphs") and self.id == 0:
            shutil.rmtree("classifiers/graphs")
            os.makedirs("classifiers/graphs")
        draw_curve(
            running_top1_accuracies,
            f"train_top1_accuracies_sgd_model_{self.id}",
            self.train_epochs,
            "Accuracy",
        )
        draw_curve(
            running_losses,
            f"train_losses_sgd_model_{self.id}",
            self.train_epochs,
            "Loss",
        )
        draw_curve(
            validation_accuracies,
            f"valid_top1_accuracies_sgd_model_{self.id}",
            self.train_epochs,
            "Accuracy",
        )
        draw_curve(
            validation_losses,
            f"valid_losses_sgd_model_{self.id}",
            self.train_epochs,
            "Loss",
        )
