import importlib
import torch
from classifiers.classifier_abc import ClassifierABC
from .algorithm_abc import AlgorithmABC


class DynamicExpertVoting(AlgorithmABC):
    NAME = "dynamic_expert_voting"

    def __init__(self, ensemble: ClassifierABC):
        super().__init__(ensemble)

    def test(
        self, dataset_name: str, cuda: bool, download_data: bool, test_batch_size: int
    ):
        """
        Tests and predicts labels using dynamic expert voting.
        """
        print()
        print("Started ensemble testing")
        dataset = importlib.import_module("dataloader.{}".format(dataset_name))
        test_loader = dataset.load_test_data(
            cuda=cuda,
            download=download_data,
            batch_size=test_batch_size,
        )
        # TODO: assuming that the first classifier is trained on all labels (ie meta classifier)
        accuracy, _, meta_confidence_scores, _ = self.ensemble[0].test(test_loader)
        print(f"meta classifier accuracy: {accuracy}")
        ensemble_predictions = torch.empty((0,))
        expert_confidence_scores = {}
        expert_member_index = 0
        for member in self.ensemble[1:]:
            test_loader = dataset.load_test_data(
                cuda=cuda,
                download=download_data,
                batch_size=test_batch_size,
                transformation_type=-1,
            )
            accuracy, _, expert_confidence_score, _ = member.test(test_loader)
            expert_confidence_scores[member] = expert_confidence_score
            print(f"expert member accuracy: {accuracy}")
            expert_member_index += 1

        targets = self.ensemble[0].get_targets()
        f = open("classifiers/outputs/confidence_scores.txt", "w")
        for i, row in enumerate(meta_confidence_scores):
            f.write("Data point: {}\n".format(i + 1))
            f.write("Model 0:\n")
            f.write("confidence score: {}\n".format(row))
            possible_classes = torch.where(row > 0.15)[0].tolist()
            if len(possible_classes) == 0:
                print(
                    "no possible classes found. Using main model confidence score's max value."
                )
                possible_class = torch.argmax(row).item()
                possible_classes = [possible_class]
            member_count = 0
            for index, member in enumerate(self.ensemble[1:]):
                if any(
                    digit in member.data_loaders.labels for digit in possible_classes
                ):
                    f.write("Model {}:\n".format(index + 1))
                    f.write("Labels: {}\n".format(member.data_loaders.labels))
                    expert_confidence_score = expert_confidence_scores[member][i]
                    f.write("confidence score: {}\n".format(expert_confidence_score))
                    if member_count == 0:
                        sum_squared_confidence_scores = torch.zeros_like(
                            expert_confidence_score
                        )
                    sum_squared_confidence_scores += expert_confidence_score**2
                    member_count += 1
            mean_squared_confidence_scores = (
                sum_squared_confidence_scores / member_count
            )
            f.write(
                "Mean squared confidence score: {}\n".format(
                    mean_squared_confidence_scores
                )
            )
            average_squared_confidence_vote = torch.argmax(
                mean_squared_confidence_scores
            )
            f.write("Ensemble vote: {}\n".format(average_squared_confidence_vote))
            ensemble_predictions = torch.cat(
                (ensemble_predictions, average_squared_confidence_vote.unsqueeze(0))
            )
            f.write("Labels: {}\n".format(targets[i]))
            f.write("\n")
        f.close()

        accuracy = self.get_accuracy(ensemble_predictions)
        print(f"dynamic expoert vote accuracy: {accuracy}")

        self.draw_test_result_by_class(ensemble_predictions, 10)
