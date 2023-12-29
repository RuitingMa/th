from typing import ClassVar, Type, Dict
from abc import ABC, abstractmethod

MODEL_REGISTRY: Dict[str, Type["ModelABC"]] = {}

__all__ = ["ModelABC", "MODEL_REGISTRY"]


class ModelABC(ABC):
    TYPE: ClassVar[str]

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "TYPE"):
            MODEL_REGISTRY[cls.TYPE] = cls

    @classmethod
    def from_config(cls, model_type) -> "ModelABC":
        try:
            model_cls = MODEL_REGISTRY[model_type]
        except KeyError:
            raise ValueError(f"{model_type} is not the name of a valid model type.")
        return model_cls()

    @abstractmethod
    def init_w(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
