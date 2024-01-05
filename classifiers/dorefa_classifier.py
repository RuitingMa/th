from typing import List, Optional
from tqdm import tqdm
from .classifier_abc import ClassifierABC, DataLoaders, ModelConfig


class DorefaClassifier(ClassifierABC):
    NAME = "dorefa"

    def __init__(
        self,
        model_config: ModelConfig,
        data_loaders: DataLoaders,
        train_epochs: Optional[int] = 100,
    ):
        super().__init__(model_config, data_loaders, train_epochs)

    def train_step(self) -> List[float]:
        losses = []
        self.model_config.model.train()

        for data, target in tqdm(
            self.data_loaders.train_loader, total=len(self.data_loaders.train_loader)
        ):
            data, target = data.to(self.model_config.device), target.to(
                self.model_config.device
            )
            self.model_config.optimizer.zero_grad()

            output = self.model_config.model(data)
            loss = self.model_config.criterion(output, target)
            losses.append(loss.item())
            loss.backward()

            self.model_config.optimizer.step()

        return losses
