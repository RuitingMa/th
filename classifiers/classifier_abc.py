from typing import ClassVar
from abc import ABC, abstractmethod
import shutil
import os
from torch import save, no_grad
from tqdm import tqdm


CLASSIFIER_REGISTRY: dict[str, type["ClassifierABC"]] = {}

__all__ = ["ClassifierABC", "CLASSIFIER_REGISTRY"]


class ClassifierABC(ABC):
    NAME: ClassVar[str]

    def __init__(self, model, train_loader=None, test_loader=None, device=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "NAME"):
            CLASSIFIER_REGISTRY[cls.NAME] = cls

    @classmethod
    def from_config(cls, name, model):
        try:
            classifier_cls = CLASSIFIER_REGISTRY[name]
        except KeyError:
            raise ValueError(f"{name} is not the name of a valid classifier type.")
        return classifier_cls(model)

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

    def test(self, criterion):
        if self.test_loader is None:
            raise ValueError(f"test loader for classifier {self} has not been set.")
        self.model.eval()
        top1 = 0
        test_loss = 0.0

        with no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                top1 += pred.eq(target.view_as(pred)).sum().item()

        top1_acc = 100.0 * top1 / len(self.test_loader.sampler)

        return top1_acc

    @abstractmethod
    def train_step(self, criterion, optimizer):
        raise NotImplementedError

    def train(self, criterion, optimizer, epochs, scheduler, checkpoint=None):
        if checkpoint is None:
            raise ValueError("Specify a valid checkpoint")
        if self.train_loader is None:
            raise ValueError("Train loader has not been specified for {self}.")

        best_accuracy = 0.0

        losses = []
        accuracies = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = self.train_step(criterion, optimizer)
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = optimizer.param_groups[0]["lr"]
            test_accuracy = self.test(criterion)
            accuracies.append(test_accuracy)
            if scheduler:
                scheduler.step()
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
                    "state_dict": self.model.state_dict(),
                    "best_accuracy": best_accuracy,
                    "optimizer": optimizer.state_dict(),
                    "criterion": criterion,
                },
                is_best,
                checkpoint,
            )
