from models.model_abc import ModelABC


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
        ensemble[model_type] = []
        for model_count in range(model_info.model.count):
            new_model = ModelABC.from_config(model_type)
            ensemble[model_type].append(new_model)
    return ensemble
