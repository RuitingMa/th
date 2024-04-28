from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Type

from matplotlib import pyplot as plt

from classifiers.classifier_abc import ClassifierABC

ALGORITHM_REGISTRY: Dict[str, Type["AlgorithmABC"]] = {}

__all__ = ["AlgorithmABC", "ALGORITHM_REGISTRY"]


class AlgorithmABC(ABC):
    """
    Represents the base class for all ensemble algorithms used to
    train/test the system.
    """

    NAME: ClassVar[str]
    """
    Name of the ensemble algorithm used to register the algorithm in the config.
    """

    def __init__(self, ensemble: List[ClassifierABC]):
        self.ensemble = ensemble

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "NAME"):
            ALGORITHM_REGISTRY[cls.NAME] = cls

    def train(self):
        """
        Trains the ensemble of classifiers by calling the train method
        of each classifier in the ensemble.
        """
        print()
        print("Started ensemble training")
        for index, classifier in enumerate(self.ensemble):
            print(f"Training classifier {index+1}")
            classifier.train()

    def get_accuracy(self, predictions: List[int]) -> float:
        """
        Returns the accuracy of the ensemble of classifiers by comparing
        the provided predictions with the test targets of the first ensemble
        member (assumes all classifiers have the same test targets (ie "all" label))
        """
        targets = self.ensemble[0].get_targets()
        return 100 * ((predictions == targets).sum().item() / len(targets))

    @abstractmethod
    def test(
        self, dataset_name: str, cuda: bool, download_data: bool, test_batch_size: int
    ):
        """
        Implements the method used by the ensemble to test the system
        and predict labels. The final predictions are inferred by
        the algorithm which the ensemble instance uses to consolidate
        individual member predictions.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, name: str, ensemble: List[ClassifierABC]) -> "AlgorithmABC":
        """
        Registers the ensemble algorithm in the config using its name
        and returns an instance of the ensemble algorithm class. Raises
        an error if the name is not valid.
        """
        try:
            ensemble_algorithm_cls = ALGORITHM_REGISTRY[name]
        except KeyError:
            raise ValueError(f"{name} is not the name of a valid ensemble algorithm.")

        return ensemble_algorithm_cls(ensemble)

    def draw_test_result_by_class(
        self, ensemble_predictions: List[int], target_length: int
    ):
        targets = self.ensemble[0].get_targets()
        incorrect_predictions_count = [0] * target_length
        correct_predictions_count = [0] * target_length
        total_count = [0] * target_length
        for i in range(len(targets)):
            total_count[targets[i]] += 1
            if targets[i] != ensemble_predictions[i]:
                incorrect_predictions_count[targets[i]] += 1
            else:
                correct_predictions_count[targets[i]] += 1
        # correct predictions
        correct_predictions = [
            correct_predictions_count[i] * 100 / total_count[i]
            for i in range(len(correct_predictions_count))
        ]
        plt.figure(figsize=(10, 6))
        plt.bar(range(target_length), correct_predictions, color="maroon", width=0.4)
        plt.xlabel("Classes")
        plt.ylabel("Percentage of correctly classified")
        plt.title("Percentage of correctly classified per class")
        plt.grid(True)
        plt.savefig(f"classifiers/graphs/correct_classification.png")
        plt.close()

        # incorrect predictions
        incorrect_predictions = [
            incorrect_predictions_count[i] * 100 / total_count[i]
            for i in range(len(incorrect_predictions_count))
        ]
        plt.figure(figsize=(10, 6))
        plt.bar(range(target_length), incorrect_predictions, color="maroon", width=0.4)
        plt.xlabel("Classes")
        plt.ylabel("Percentage of oncorrectly classified")
        plt.title("Percentage of incorrectly classified per class")
        plt.grid(True)
        plt.savefig(f"classifiers/graphs/incorrect_classification.png")
        plt.close()
