from .algorithm_abc import AlgorithmABC


class Bagging(AlgorithmABC):
    def __init__(self, ensemble):
        super().__init__(ensemble)

    def test(ensemble):
        raise NotImplementedError
