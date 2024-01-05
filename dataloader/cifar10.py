import os
from typing import List
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset, ConcatDataset
import torchvision.transforms as tvt


def load_train_data(
    labels: List[int] = None,
    batch_size: int = 64,
    sampler: torch.utils.data.Sampler = None,
    cuda: bool = False,
    enable_transform: bool = True,
    download: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Loads training data for CIFAR10 dataset using the specified options.

    Args:
        labels: List of labels to load. If None, all labels are loaded.
        batch_size: Batch size.
        sampler: Sampler to use for loading data.
        cuda: Whether to use CUDA.
        enable_transform: Whether to enable data augmentation.
        download: Whether to download the dataset.

    Returns:
        DataLoader for the training data.
    """
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
        indices = [i for i, x in enumerate(dataset.targets) if x in labels]
        target_set = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            dataset=target_set,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **loader_kwargs,
        )
    return loader


def load_test_data(
    labels: List[int] = None,
    batch_size: int = 1000,
    sampler: torch.utils.data.Sampler = None,
    cuda: bool = False,
    download: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Loads test data for CIFAR10 dataset using the specified options.

    Args:
        labels: List of labels to load. If None, all labels are loaded.
        batch_size: Batch size.
        sampler: Sampler to use for loading data.
        cuda: Whether to use CUDA.
        download: Whether to download the dataset.

    Returns:
        DataLoader for the test data.
    """
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
        indices = [i for i, x in enumerate(dataset.targets) if x in labels]
        target_set = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            dataset=target_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            **loader_kwargs,
        )

    return loader
