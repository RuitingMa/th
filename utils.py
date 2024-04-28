import importlib
import os
from typing import List

from models.model_abc import ModelABC
from classifiers.classifier_abc import ClassifierABC


def build_ensemble(
    ensemble_info: List[ClassifierABC],
    device: str,
    dataset_name: str,
    cuda: bool,
    download_data: bool,
) -> List[ClassifierABC]:
    """
    Builds the ensemble of classifiers using the config information.

    Returns a list containing the classifiers.
    """
    ensemble = []
    dataset = importlib.import_module("dataloader.{}".format(dataset_name))
    transformation_type_index = 0
    id = 0
    for model_info in ensemble_info:
        model_type = model_info.model.model_type
        bin_type = model_info.model.bin_type
        optimizer_flag = model_info.model.optimizer
        learning_rate = model_info.model.lr
        steps = model_info.model.steps
        gamma = model_info.model.gamma
        epochs = model_info.model.epochs
        checkpoint = model_info.model.checkpoint
        test_batch_size = model_info.model.test_batch_size
        train_batch_size = model_info.model.train_batch_size
        # TODO: add enable_transformation flag control
        for model_count in range(model_info.model.count):
            new_model = ModelABC.from_config(model_type)
            model_checkpoint = os.path.join(checkpoint, f"model_{model_count}")
            labels = model_info.model.train_labels[model_count]
            print(f"labels: {labels}")
            if labels == "all":
                train_loader = dataset.load_train_data(
                    cuda=cuda, download=download_data, batch_size=train_batch_size
                )
                test_loader = dataset.load_test_data(
                    cuda=cuda,
                    download=download_data,
                    batch_size=test_batch_size,
                )
            else:
                train_loader = dataset.load_train_data(
                    labels=labels,
                    cuda=cuda,
                    download=download_data,
                    batch_size=train_batch_size,
                    transformation_type=-1,
                )
                test_loader = dataset.load_test_data(
                    labels=labels,
                    cuda=cuda,
                    download=download_data,
                    batch_size=test_batch_size,
                    transformation_type=-1,
                )
                transformation_type_index += 1
            classifier = ClassifierABC.from_config(
                bin_type,
                new_model,
                optimizer_flag,
                learning_rate,
                steps,
                gamma,
                model_checkpoint,
                id,
                labels,
                epochs,
                train_loader,
                test_loader,
                device,
            )
            ensemble.append(classifier)
            id += 1
    return ensemble
