import torch
import importlib
from classifiers.xnor_classifier import *
from models import *
from BENN import *


class Model:
    def __init__(self) -> None:
        pass


class Ensemble:
    def __init__(self, ensemble) -> None:
        pass

    # def __init__(
    #     self,
    #     bin_type_1,
    #     bin_type_2,
    #     dataset,
    #     train_batch_size_1,
    #     train_batch_size_2,
    #     test_batch_size,
    #     model_1,
    #     model_2,
    #     device,
    #     optimizer,
    #     lr,
    #     steps,
    #     gamma,
    # ):
    #     self.model_1 = model_1
    #     self.model_2 = model_2
    #     self.device = device
    #     self.dataset = importlib.import_module("dataloader.{}".format(dataset))
    #     # parition for the members of the ensemble
    #     train_loader_0 = self.dataset.load_all_train_data(train_batch_size_2)
    #     train_data = self.dataset.load_train_data("intelligent", train_batch_size_1)

    #     # test data set
    #     self.test_loader = self.dataset.load_test_data(test_batch_size)
    #     # Only XNOR has been used in the study so far
    #     # but you can use DoReFa and BNN in the same manner as well as they have been implemented
    #     self.model0 = None
    #     self.expert_classifiers = []
    #     # models 1-6
    #     if bin_type_1 == "xnor":
    #         classification_1 = XnorClassifier(
    #             self.model_1, train_data[0], self.test_loader, self.device
    #         )
    #         classification_2 = XnorClassifier(
    #             self.model_1, train_data[1], self.test_loader, self.device
    #         )
    #         classification_3 = XnorClassifier(
    #             self.model_1, train_data[2], self.test_loader, self.device
    #         )
    #         classification_4 = XnorClassifier(
    #             self.model_1, train_data[3], self.test_loader, self.device
    #         )

    #         # classification_5 = XnorClassifier(self.model_1, train_data[4], self.test_loader, self.device)
    #         # classification_6 = XnorClassifier(self.model_1, train_data[5], self.test_loader, self.device)
    #         # CIFAR-10 partition
    #         classification_1.set_label([4, 9, 7, 1])
    #         classification_2.set_label([5, 8, 3, 2])
    #         classification_3.set_label([0, 6, 5, 3])

    #         classification_4.set_label([1, 2, 8, 7])

    #         # classification_5.set_label([7,2,4,3])
    #         # classification_6.set_label([7,0,3,8])

    #         self.expert_classifiers.append(classification_1)
    #         self.expert_classifiers.append(classification_2)
    #         self.expert_classifiers.append(classification_3)
    #         self.expert_classifiers.append(classification_4)
    #         # self.expert_classifiers.append(classification_5)
    #         # self.expert_classifiers.append(classification_6)

    #     # model 0
    #     if bin_type_2 == "xnor":
    #         classification_0 = XnorClassifier(
    #             model_2, train_loader_0, self.test_loader, self.device
    #         )
    #         self.model0 = classification_0

    #     self.criterion = torch.nn.CrossEntropyLoss()
    #     self.criterion.to(device)

    #     # initialising model weights
    #     if hasattr(self.model_1, "init_w"):
    #         self.model_1.init_w()

    #     if hasattr(self.model_2, "init_w"):
    #         self.model_2.init_w()

    #     # optimiser
    #     self.optimizer_1 = None
    #     self.optimizer_2 = None
    #     if optimizer == "adam":
    #         self.optimizer_1 = torch.optim.Adam(
    #             self.model_1.parameters(), lr=lr, weight_decay=1e-5
    #         )
    #         self.optimizer_2 = torch.optim.Adam(
    #             self.model_2.parameters(), lr=lr, weight_decay=1e-5
    #         )
    #     elif optimizer == "sgd":
    #         self.optimizer_1 = torch.optim.SGD(
    #             self.model_1.parameters(), lr=lr, momentum=0.9, weight_decay=5.0e-4
    #         )
    #         self.optimizer_2 = torch.optim.SGD(
    #             self.model_2.parameters(), lr=lr, momentum=0.9, weight_decay=5.0e-4
    #         )

    #     self.scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(
    #         self.optimizer_1, steps, gamma=gamma
    #     )
    #     self.scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(
    #         self.optimizer_2, steps, gamma=gamma
    #     )

    # def train(self, checkpoint, epochs_1, epochs_2):
    #     preds_0 = None
    #     for member in self.expert_classifiers:
    #         test_accuracy, preds, conf = member.train(
    #             self.criterion, self.optimizer_1, epochs_1, self.scheduler_1, checkpoint
    #         )

    #     test_accuracy, preds, conf = self.model0.train(
    #         self.criterion, self.optimizer_2, epochs_2, self.scheduler_2, checkpoint
    #     )
    #     preds_0 = preds
    #     self.model0.confidences = conf

    #     return preds_0

    # def test(self, preds_0):
    #     benn = BENN(self.device, self.test_loader)
    #     preds_ensemble = benn.square_confidence_modified(
    #         self.expert_classifiers, self.model0.confidences, preds_0
    #     )
