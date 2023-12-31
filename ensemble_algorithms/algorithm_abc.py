from abc import ABC, abstractmethod
from typing import ClassVar


class AlgorithmABC:
    NAME: ClassVar[str]

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def train(self):
        for classifier in self.ensemble:
            classifier.train()

    # @abstractmethod
    # def test(ensemble):
    #     raise NotImplementedError

    # @abstractmethod
    # def predict(ensemble):
    #     raise NotImplementedError
