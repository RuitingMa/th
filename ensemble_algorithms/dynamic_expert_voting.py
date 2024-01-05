import torch
from classifiers.classifier_abc import ClassifierABC
from .algorithm_abc import AlgorithmABC


class DynamicExpertVoting(AlgorithmABC):
    NAME = "dynamic_expert_vote"

    def __init__(self, ensemble: ClassifierABC):
        super().__init__(ensemble)

    def test(self):
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
            member_count = 0
            for member in self.ensemble[1:]:
                # print("indices vs digits")
                # print(
                #     possible_classes, member.data_loaders.train_loader.dataset.indices
                # )
                if any(
                    digit in member.data_loaders.train_loader.dataset.indices
                    for digit in possible_classes
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
            # print("mean squared confidence scores")
            # print(mean_squared_confidence_scores)
            average_squared_confidence_vote = torch.argmax(
                mean_squared_confidence_scores
            )
            # print(average_squared_confidence_vote)
            ensemble_predictions = torch.cat(
                (ensemble_predictions, average_squared_confidence_vote.unsqueeze(0))
            )
            # print(ensemble_predictions)
        accuracy = self.get_accuracy(ensemble_predictions)
        print(f"dynamic expoert vote accuracy: {accuracy}")
