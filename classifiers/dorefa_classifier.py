import torch
from typing import List, Optional, Tuple
from tqdm import tqdm
from .classifier_abc import ClassifierABC, DataLoaders, ModelConfig

class DorefaClassifier(ClassifierABC):
    """
    Represents a Dorefa classifier.
    """

    NAME = "dorefa"

    def __init__(
        self,
        id: int,
        model_config: ModelConfig,
        data_loaders: DataLoaders,
        train_epochs: Optional[int] = 100,
    ):
        super().__init__(id=id, model_config=model_config, data_loaders=data_loaders, train_epochs=train_epochs)

    def train_step(self) -> Tuple[List[float], float]:
        losses = []
        correct = 0
        total = 0
        self.model_config.model.train()

        for data, target in tqdm(
            self.data_loaders.train_loader, total=len(self.data_loaders.train_loader)
        ):
            data, target = data.to(self.model_config.device), target.to(self.model_config.device)
            self.model_config.optimizer.zero_grad()

            output = self.model_config.model(data)
            loss = self.model_config.criterion(output, target)
            losses.append(loss.item())

            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            loss.backward()
            self.model_config.optimizer.step()

        epoch_accuracy = 100 * correct / total  # 计算整个epoch的准确率
        return losses, epoch_accuracy
