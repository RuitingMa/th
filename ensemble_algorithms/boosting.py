import torch
from classifiers.classifier_abc import ClassifierABC
from .algorithm_abc import AlgorithmABC


class Boosting(AlgorithmABC):
    NAME = "boosting"

    def __init__(self, ensemble: ClassifierABC):
        super().__init__(ensemble)

    def train(self):
        """
        Trains the ensemble using boosting.
        """
        print()
        print("Started ensemble training")
        for index, classifier in enumerate(self.ensemble):
            print(f"Training classifier {index+1}")
            classifier.train()
