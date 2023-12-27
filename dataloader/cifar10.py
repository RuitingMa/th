import os
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset, ConcatDataset
import torchvision.transforms as tvt


def load_train_data(
    labels=None,
    batch_size=64,
    sampler=None,
    cuda=False,
    enable_transform=True,
    download=False,
):
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}
    transform = tvt.Compose(
        [
            tvt.RandomCrop(32, padding=4),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor(),
            tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = CIFAR10(
        os.path.join("datasets", "cifar10"), train=True, download=download
    )
    if sampler is None:
        shuffle = True
    else:
        shuffle = False
    if enable_transform:
        dataset.transform = transform
    if labels is None:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        target_set = torch.utils.data.Subset(dataset, labels)
        loader = torch.utils.data.DataLoader(
            dataset=target_set,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **loader_kwargs,
        )
    return loader


def load_test_data(
    labels=None,
    batch_size=1000,
    sampler=None,
    cuda=False,
    download=False,
):
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}

    transform = tvt.Compose(
        [
            tvt.RandomCrop(32, padding=4),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor(),
            tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset = CIFAR10(
        os.path.join("datasets", "cifar10"),
        train=False,
        download=download,
        transform=transform,
    )
    if labels is None:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        target_set = torch.utils.data.Subset(dataset, labels)
        loader = torch.utils.data.DataLoader(
            dataset=target_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            **loader_kwargs,
        )

    return loader
