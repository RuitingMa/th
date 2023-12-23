from typing import ClassVar

MODEL_REGISTRY: dict[str, type["ModelABC"]] = {}

__all__ = ["ModelABC"]


class ModelABC:
    TYPE: ClassVar[str]

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "TYPE"):
            MODEL_REGISTRY[cls.TYPE] = cls

    @classmethod
    def from_config(cls, type):
        try:
            model_cls = MODEL_REGISTRY[type]
        except KeyError:
            raise ValueError(f"{type} is not the name of a valid model type.")
        return model_cls()
