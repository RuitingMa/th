from abc import ABC, abstractmethod
from typing import ClassVar


class AlgorithmABC:
    NAME: ClassVar[str]

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def train(self):
        print()
        print("Started ensemble training")
        for index, classifier in enumerate(self.ensemble):
            print(f"Training classifier {index+1}")
            classifier.train()

    def get_accuracy(self, predictions):
        # assumes all models have the same test targets (ie "all" labels)
        targets = self.ensemble[0].get_targets()
        return 100 * ((predictions == targets).sum().item() / len(targets))

    # @abstractmethod
    # def test(ensemble):
    #     raise NotImplementedError

    # @abstractmethod
    # def predict(ensemble):
    #     raise NotImplementedError
