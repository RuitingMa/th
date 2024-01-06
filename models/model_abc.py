from typing import ClassVar, Type, Dict
from abc import ABC, abstractmethod

import torch

MODEL_REGISTRY: Dict[str, Type["ModelABC"]] = {}

__all__ = ["ModelABC", "MODEL_REGISTRY"]


class ModelABC(ABC):
    """
    Represents the base class for all models used to train/test the classifier.
    These are essentianlly the building blocks of the ensemble.
    """

    TYPE: ClassVar[str]
    """
    Type(name) of the model used to register the model in the config.
    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "TYPE"):
            MODEL_REGISTRY[cls.TYPE] = cls

    @classmethod
    def from_config(cls, model_type) -> "ModelABC":
        """
        Retrieves and instance of the model class from the registry using
        the model type. Raises an error if the model type is not valid.

        Returns:
            Instance of the model class.
        """
        try:
            model_cls = MODEL_REGISTRY[model_type]
        except KeyError:
            raise ValueError(f"{model_type} is not the name of a valid model type.")
        return model_cls()

    @abstractmethod
    def init_w(self):
        """
        Initializes the weights of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass of the model.
        """
        raise NotImplementedError
