import importlib
from models.model_abc import ModelABC
from classifiers.classifier_abc import ClassifierABC


def build_ensemble(ensemble_info: list) -> dict[str, list[ModelABC]]:
    """
    Builds the ensemble of models using the config information.

    Returns a dictionary containing the model type as key and a list
    of models for the key as values. The number of items (models) in the list
    depends on the value of the count option passed in the config.
    """
    ensemble = {}
    for model_info in ensemble_info:
        model_type = model_info.model.model_type
        bin_type = model_info.model.bin_type
        ensemble[bin_type] = []
        for model_count in range(model_info.model.count):
            new_model = ModelABC.from_config(model_type)
            ensemble[bin_type].append(
                (new_model, model_info.model.train_labels[model_count])
            )
    return ensemble


def attach_dataset_to_ensemble(ensemble, dataset_name, device):
    """
    Attaches dataset to the ensemble of models (ie making an ensemble of classifiers)
    """
    classifiers = []
    dataset = importlib.import_module("dataloader.{}".format(dataset_name))
    for bin_type, (model, labels) in ensemble.items():
        classifier = ClassifierABC.from_config(bin_type, model)
        if labels == "all":
            classifier.train_loader = dataset.load_all_train_data()
            classifier.test_loader = dataset.load_test_data()
            classifier.device = device
        else:
            classifier.train_loader = dataset.load_train_data(labels)
            classifier.test_loader = dataset.load_test_data(labels)
            classifier.device = device
        classifiers.append(classifier)
    return classifiers
