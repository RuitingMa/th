import importlib
import os
from typing import List
from models.model_abc import ModelABC
from classifiers.classifier_abc import ClassifierABC


def build_ensemble(
    ensemble_info: list,
    device,
    dataset_name,
    cuda,
    download_data,
) -> List[ClassifierABC]:
    """
    Builds the ensemble of classifiers using the config information.

    Returns a dictionary containing the model type as key and a list
    of models for the key as values. The number of items (models) in the list
    depends on the value of the count option passed in the config.
    """
    ensemble = []
    dataset = importlib.import_module("dataloader.{}".format(dataset_name))
    for model_info in ensemble_info:
        model_type = model_info.model.model_type
        bin_type = model_info.model.bin_type
        optimizer_flag = model_info.model.optimizer
        learning_rate = model_info.model.lr
        steps = model_info.model.steps
        gamma = model_info.model.gamma
        epochs = model_info.model.epochs
        checkpoint = model_info.model.checkpoint
        for model_count in range(model_info.model.count):
            new_model = ModelABC.from_config(model_type)
            model_checkpoint = os.path.join(checkpoint, f"model_{model_count}")
            labels = model_info.model.train_labels[model_count]
            if labels == "all":
                train_loader = dataset.load_train_data(
                    cuda=cuda, download=download_data
                )
                test_loader = dataset.load_test_data(cuda=cuda, download=download_data)
            else:
                train_loader = dataset.load_train_data(
                    labels=labels, cuda=cuda, download=download_data
                )
                test_loader = dataset.load_test_data(
                    labels=labels, cuda=cuda, download=download_data
                )
            classifier = ClassifierABC.from_config(
                bin_type,
                new_model,
                optimizer_flag,
                learning_rate,
                steps,
                gamma,
                model_checkpoint,
                epochs,
                train_loader,
                test_loader,
                device,
            )
            ensemble.append(classifier)
    return ensemble
