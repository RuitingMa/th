import torchvision.transforms as tvt
import os
from torchvision import datasets
import torch
from typing import List


def load_train_data(
    labels: List[int] = None,
    batch_size: int = 64,
    sampler: torch.utils.data.Sampler = None,
    cuda: bool = False,
    enable_transform: bool = True,
    download: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Args:
        enable_transform: Whether to enable data augmentation.
        cuda: Whether to use CUDA (GPU acceleration).
        sampler: Optional PyTorch sampler for the DataLoader.

    Returns:
        DataLoader for the training data.
    """
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}
    transform_list = [
        tvt.ToTensor(),
        tvt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    if enable_transform:
        transform_list = [
            tvt.RandomResizedCrop(224),
            tvt.RandomHorizontalFlip(),
        ] + transform_list
    transform = tvt.Compose(transform_list)
    dataset = datasets.ImageFolder(
        os.path.join("path_to_imagenet", "train"), transform=transform
    )
    shuffle = sampler is None
    return torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=shuffle, sampler=sampler, **loader_kwargs
    )
