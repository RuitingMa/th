import importlib
from typing import Type
from models.model_abc import ModelABC
from classifiers.classifier_abc import ClassifierABC


def build_ensemble(
    ensemble_info: list,
    device,
    dataset_name,
    cuda,
    download_data,
) -> list[ClassifierABC]:
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
        for model_count in range(model_info.model.count):
            new_model = ModelABC.from_config(model_type)
            classifier = ClassifierABC.from_config(bin_type, new_model)
            labels = model_info.model.train_labels[model_count]
            if labels == "all":
                classifier.train_loader = dataset.load_train_data(
                    cuda=cuda, download=download_data
                )
                classifier.test_loader = dataset.load_test_data(
                    cuda=cuda, download=download_data
                )
                classifier.device = device
            else:
                classifier.train_loader = dataset.load_train_data(
                    labels=labels, cuda=cuda, download=download_data
                )
                classifier.test_loader = dataset.load_test_data(
                    labels=labels, cuda=cuda, download=download_data
                )
                classifier.device = device
            ensemble.append(classifier)
    return ensemble
