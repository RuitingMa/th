import torch
from os.path import join
from torchvision.datasets import MNIST
import torchvision.transforms as tvt


def load_train_data(
    labels=None,
    batch_size=64,
    sampler=None,
    cuda=True,
    enable_transform=True,
    download=False,
):
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}
    transform = tvt.Compose(
        [
            tvt.Resize((28, 28)),
            tvt.ToTensor(),
            tvt.Normalize((0.1307,), (0.308,)),
        ]
    )
    dataset = MNIST(join("datasets", "mnist"), train=True, download=download)
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


def load_test_data(batch_size=1000, sampler=None, cuda=False, download=False):
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}

    transform = tvt.Compose(
        [
            tvt.Resize((28, 28)),
            tvt.ToTensor(),
            tvt.Normalize((0.1307,), (0.308,)),
        ]
    )

    dataset = MNIST(
        join("datasets", "mnist"),
        train=False,
        download=download,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler, **loader_kwargs
    )

    return loader
