import torch
from classifiers.classifier_abc import ClassifierABC
from .algorithm_abc import AlgorithmABC


class Bagging(AlgorithmABC):
    NAME = "bagging"

    def __init__(self, ensemble: ClassifierABC):
        super().__init__(ensemble)

    def test(self):
        self._hard_voting()

    def _hard_voting(self):
        all_predictions = torch.empty((0,))
        for member in self.ensemble:
            accuracy, predictions = member.test()
            predictions = predictions.view(-1, 1)
            all_predictions = torch.cat((all_predictions, predictions), dim=1)
            print(f"member accuracy: {accuracy}")
        majority_votes, _ = torch.mode(all_predictions, dim=1)
        accuracy = self._get_accuracy(majority_votes)
        print(f"hard voting accuracy: {accuracy}")
        return majority_votes

    def _get_accuracy(self, predictions):
        targets = self.ensemble[0].get_targets()
        return 100 * ((predictions == targets).sum().item() / len(targets))
