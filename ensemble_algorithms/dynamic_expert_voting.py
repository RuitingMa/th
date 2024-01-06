import torch
from classifiers.classifier_abc import ClassifierABC
from .algorithm_abc import AlgorithmABC


class DynamicExpertVoting(AlgorithmABC):
    NAME = "dynamic_expert_voting"

    def __init__(self, ensemble: ClassifierABC):
        super().__init__(ensemble)

    def test(self):
        """
        Tests and predicts labels using dynamic expert voting.
        """
        print()
        print("Started ensemble testing")
        # TODO: assuming that the first classifier is trained on all labels (ie meta classifier)
        accuracy, _, meta_confidence_scores = self.ensemble[0].test()
        print(f"meta classifier accuracy: {accuracy}")
        ensemble_predictions = torch.empty((0,))
        expert_confidence_scores = {}
        for member in self.ensemble[1:]:
            accuracy, _, expert_confidence_score = member.test()
            expert_confidence_scores[member] = expert_confidence_score
            print(f"expert member accuracy: {accuracy}")

        for i, row in enumerate(meta_confidence_scores):
            possible_classes = torch.where(row > 0.15)[0].tolist()
            if len(possible_classes) == 0:
                print(
                    "no possible classes found. Using main model confidence score's max value."
                )
                possible_class = torch.argmax(row).item()
                possible_classes = [possible_class]
            member_count = 0
            for member in self.ensemble[1:]:
                if any(
                    digit in member.data_loaders.labels for digit in possible_classes
                ):
                    expert_confidence_score = expert_confidence_scores[member][i]
                    if member_count == 0:
                        sum_squared_confidence_scores = torch.zeros_like(
                            expert_confidence_score
                        )
                    sum_squared_confidence_scores += expert_confidence_score**2
                    member_count += 1
            mean_squared_confidence_scores = (
                sum_squared_confidence_scores / member_count
            )
            average_squared_confidence_vote = torch.argmax(
                mean_squared_confidence_scores
            )
            ensemble_predictions = torch.cat(
                (ensemble_predictions, average_squared_confidence_vote.unsqueeze(0))
            )
        accuracy = self.get_accuracy(ensemble_predictions)
        print(f"dynamic expoert vote accuracy: {accuracy}")
