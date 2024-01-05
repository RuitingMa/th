from typing import List, Optional
from tqdm import tqdm
from .classifier_abc import ClassifierABC, DataLoaders, ModelConfig


class BnnClassifier(ClassifierABC):
    NAME = "bnn"

    def __init__(
        self,
        model_config: ModelConfig,
        data_loaders: DataLoaders,
        train_epochs: Optional[int] = 100,
    ):
        super().__init__(model_config, data_loaders, train_epochs)

    def train_step(self) -> List[float]:
        losses = []
        for data, target in tqdm(
            self.data_loaders.train_loader, total=len(self.data_loaders.train_loader)
        ):
            data, target = data.to(self.model_config.device), target.to(
                self.model_config.device
            )
            output = self.model_config.model(data)
            loss = self.model_config.criterion(output, target)
            losses.append(loss.item())
            self.model_config.optimizer.zero_grad()
            loss.backward()
            for p in self.model_config.model.modules():
                if hasattr(p, "weight_org"):
                    p.weight.data.copy_(p.weight_org)
            self.model_config.optimizer.step()
            for p in self.model_config.model.modules():
                if hasattr(p, "weight_org"):
                    p.weight_org.data.copy_(p.weight.data.clamp_(-1, 1))
        return losses
