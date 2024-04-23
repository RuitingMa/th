from typing import List, Optional
from tqdm import tqdm
from models.xnor_layers import XNORConv2d
from .classifier_abc import ClassifierABC, DataLoaders, ModelConfig


class XnorClassifier(ClassifierABC):
    """
    Represents a XNOR classifier.
    """

    NAME = "xnor"

    def __init__(
        self,
        id: int,
        model_config: ModelConfig,
        data_loaders: DataLoaders,
        train_epochs: Optional[int] = 100,
    ):
        super().__init__(id, model_config, data_loaders, train_epochs)

    def train_step(self) -> List[float]:
        losses = []
        top1 = 0
        self.model_config.model.train()

        for data, target in tqdm(
            self.data_loaders.train_loader, total=len(self.data_loaders.train_loader)
        ):
            data, target = data.to(self.model_config.device), target.to(
                self.model_config.device
            )
            self.model_config.optimizer.zero_grad()

            output = self.model_config.model(data)
            pred = output.argmax(dim=1, keepdim=True)
            top1 += pred.eq(target.view_as(pred)).sum().item()
            top1_acc = 100.0 * top1 / len(self.data_loaders.train_loader.sampler)
            loss = self.model_config.criterion(output, target)
            losses.append(loss.item())
            loss.backward()

            for m in self.model_config.model.modules():
                if isinstance(m, XNORConv2d):
                    m.update_gradient()

            self.model_config.optimizer.step()

        return losses, top1_acc
