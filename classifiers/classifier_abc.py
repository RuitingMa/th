from typing import ClassVar


class ClassifierABC:
    name: ClassVar[str]
    model: ClassVar[ModelABC]

    def __init__(self, name, model):
        self.name = name
        self.model = model
