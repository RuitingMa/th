import torch
from classifiers.classifier_abc import ClassifierABC
from .algorithm_abc import AlgorithmABC


class Bagging(AlgorithmABC):
    NAME = "bagging"

    def __init__(self, ensemble: ClassifierABC):
        super().__init__(ensemble)

    def test(self) -> float:
        """
        Tests the ensemble and calculates system accuracy using hard voting,
        average confidence scores, and average squared confidence scores.
        """
        print()
        print("Started ensemble testing")
        all_predictions = torch.empty((0,))
        for i, member in enumerate(self.ensemble):
            accuracy, predictions, confidence_scores = member.test()
            if i == 0:
                sum_confidence_scores = torch.zeros_like(confidence_scores)
                sum_squared_confidence_scores = torch.zeros_like(confidence_scores)
            predictions = predictions.view(-1, 1)
            sum_confidence_scores += confidence_scores
            sum_squared_confidence_scores += confidence_scores**2
            all_predictions = torch.cat((all_predictions, predictions), dim=1)
            print(f"member accuracy: {accuracy}")
        # hard voting
        print()
        majority_votes, _ = torch.mode(all_predictions, dim=1)
        accuracy = self.get_accuracy(majority_votes)
        print(f"hard voting accuracy: {accuracy}")
        # average confidence scores
        mean_confidence_scores = sum_confidence_scores / len(self.ensemble)
        _, average_confidence_vote = torch.max(mean_confidence_scores, dim=1)
        accuracy = self.get_accuracy(average_confidence_vote)
        print(f"average confidence accuracy: {accuracy}")
        # average confidence scores
        mean_squared_confidence_scores = sum_squared_confidence_scores / len(
            self.ensemble
        )
        _, average_squared_confidence_vote = torch.max(
            mean_squared_confidence_scores, dim=1
        )
        accuracy = self.get_accuracy(average_squared_confidence_vote)
        print(f"average squared confidence accuracy: {accuracy}")
