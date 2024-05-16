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
        print("Started ensemble testing")
        dataset = importlib.import_module("dataloader.{}".format(dataset_name))
        test_loader = dataset.load_test_data(
            cuda=cuda,
            download=download_data,
            batch_size=test_batch_size,
        )
        # Assuming that the first classifier is trained on all labels (ie meta classifier)
        accuracy, _, meta_confidence_scores, _ = self.ensemble[0].test(test_loader)
        print(f"meta classifier accuracy: {accuracy}")
        
        device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
        ensemble_predictions = torch.empty((0,), device=device)
        expert_confidence_scores = {}
        
        # Testing expert members
        for index, member in enumerate(self.ensemble[1:], start=1):
            test_loader = dataset.load_test_data(
                cuda=cuda,
                download=download_data,
                batch_size=test_batch_size,
                transformation_type=index - 1,
            )
            accuracy, _, expert_confidence_score, _ = member.test(test_loader)
            expert_confidence_scores[member] = expert_confidence_score.to(device)
            print(f"expert member accuracy: {accuracy}")
        
        # Writing confidence scores and predictions to file
        with open("classifiers/outputs/confidence_scores.txt", "w") as f:
            for i, row in enumerate(meta_confidence_scores):
                f.write(f"Data point: {i + 1}\nModel 0:\nconfidence score: {row}\n")
                possible_classes = torch.where(row > 0.15)[0].tolist()
                if not possible_classes:
                    # print("No possible classes found. Using main model confidence score's max value.")
                    possible_classes = [torch.argmax(row).item()]
                
                sum_squared_confidence_scores = torch.zeros_like(row, device=device)
                member_count = 0
                for member in self.ensemble[1:]:
                    if any(digit in member.data_loaders.labels for digit in possible_classes):
                        expert_confidence_score = expert_confidence_scores[member][i]
                        f.write(f"Model {member_count + 1}:\n")
                        f.write(f"Labels: {member.data_loaders.labels}\n")
                        f.write(f"confidence score: {expert_confidence_score}\n")
                        sum_squared_confidence_scores += expert_confidence_score ** 2
                        member_count += 1
                
                if member_count > 0:
                    mean_squared_confidence_scores = sum_squared_confidence_scores / member_count
                    average_squared_confidence_vote = torch.argmax(mean_squared_confidence_scores).item()
                    ensemble_predictions = torch.cat(
                        (ensemble_predictions, torch.tensor([average_squared_confidence_vote], device=device))
                    )
                    f.write(f"Mean squared confidence score: {mean_squared_confidence_scores}\n")
                    f.write(f"Ensemble vote: {average_squared_confidence_vote}\n")
        
        # Final accuracy calculation and drawing results
        targets = self.ensemble[0].get_targets()  # Assume get_targets is implemented
        accuracy = self.get_accuracy(ensemble_predictions, targets)
        print(f"dynamic expert vote accuracy: {accuracy}")
        self.draw_test_result_by_class(ensemble_predictions, targets)

    def get_accuracy(self, predictions, targets):
        return (predictions == targets).float().mean().item() * 100

    def draw_test_result_by_class(self, predictions, targets):
        # Implement visualization or further analysis of test results by class
        pass
