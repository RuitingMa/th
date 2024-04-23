import os
from typing import List
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as tvt

PERSPECTIVE_TRANSFORMATIONS = [
    ([[0, 0], [31, 0], [31, 31], [0, 31]], [[3, 9], [25, 6], [27, 27], [2, 25]]),
    ([[0, 0], [31, 0], [31, 31], [0, 31]], [[7, 9], [30, 9], [23, 30], [8, 23]]),
    ([[0, 0], [31, 0], [31, 31], [0, 31]], [[3, 7], [22, 1], [23, 31], [4, 24]]),
    ([[0, 0], [31, 0], [31, 31], [0, 31]], [[7, 7], [23, 7], [29, 27], [4, 28]]),
]


class CustomPerspectiveTransform:
    def __init__(self, transformation_type):
        self.startpoints = PERSPECTIVE_TRANSFORMATIONS[transformation_type][0]
        self.endpoints = PERSPECTIVE_TRANSFORMATIONS[transformation_type][1]

    def __call__(self, img):
        img = tvt.functional.perspective(
            img,
            startpoints=self.startpoints,
            endpoints=self.endpoints,
            interpolation=tvt.functional.InterpolationMode.BILINEAR,
        )
        return img


def load_train_data(
    labels: List[int] = None,
    batch_size: int = 64,
    sampler: torch.utils.data.Sampler = None,
    cuda: bool = False,
    enable_transform: bool = True,
    download: bool = False,
    transformation_type: int = -1,
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
        transformation_type: Type of perspective transformation to apply. This
        is only relevant when labels is not None.

    Returns:
        DataLoader for the training data.
    """
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}
    dataset = CIFAR10(
        os.path.join("datasets", "cifar10"), train=True, download=download
    )
    if sampler is None:
        shuffle = True
    else:
        shuffle = False
    if labels is None:
        transform = tvt.Compose(
            [
                tvt.RandomCrop(32, padding=4),
                tvt.RandomHorizontalFlip(),
                tvt.ToTensor(),
                tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        if enable_transform:
            dataset.transform = transform
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        if transformation_type == -1:
            transform = tvt.Compose(
                [
                    tvt.RandomCrop(32, padding=4),
                    tvt.RandomHorizontalFlip(),
                    tvt.ToTensor(),
                    tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        else:
            transform = tvt.Compose(
                [
                    CustomPerspectiveTransform(transformation_type),
                    tvt.RandomCrop(32, padding=4),
                    tvt.RandomHorizontalFlip(),
                    tvt.ToTensor(),
                    tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        if enable_transform:
            dataset.transform = transform
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
    transformation_type: int = -1,
) -> torch.utils.data.DataLoader:
    """
    Loads test data for CIFAR10 dataset using the specified options.

    Args:
        labels: List of labels to load. If None, all labels are loaded.
        batch_size: Batch size.
        sampler: Sampler to use for loading data.
        cuda: Whether to use CUDA.
        download: Whether to download the dataset.
        transformation_type: Type of perspective transformation to apply. This
        is only relevant when labels is not None.

    Returns:
        DataLoader for the test data.
    """
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}

    dataset = CIFAR10(
        os.path.join("datasets", "cifar10"),
        train=False,
        download=download,
    )
    # load full test data with no perspective transformation
    if labels is None and transformation_type == -1:
        transform = tvt.Compose(
            [
                tvt.Resize((32, 32)),
                tvt.ToTensor(),
                tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        dataset.transform = transform
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            **loader_kwargs,
        )
    # load full test data with perspective transformation
    elif labels is None and transformation_type != -1:
        transform = tvt.Compose(
            [
                CustomPerspectiveTransform(transformation_type),
                tvt.Resize((32, 32)),
                tvt.ToTensor(),
                tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        dataset.transform = transform
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            **loader_kwargs,
        )
    # load partial test data with perspective transformation
    elif labels is not None:
        if transformation_type != -1:
            transform = tvt.Compose(
                [
                    CustomPerspectiveTransform(transformation_type),
                    tvt.Resize((32, 32)),
                    tvt.ToTensor(),
                    tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        else:
            transform = tvt.Compose(
                [
                    tvt.Resize((32, 32)),
                    tvt.ToTensor(),
                    tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        dataset.transform = transform
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
