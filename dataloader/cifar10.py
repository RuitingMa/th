import os
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset, ConcatDataset
import torchvision.transforms as tvt


def load_train_data(partition_mode,batch_size=64, sampler=None):
    cuda = True
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

    expert_transform = tvt.Compose([
        tvt.RandomCrop(32, padding=4),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True,
            download=False)

    if sampler is None:
        shuffle = True
    else:
        shuffle = False

    data = []
    # load the intellignet partitioning mode
    if (partition_mode == "intellignet"):
        for j in range(0,6):
            indices = []
            if (j==0):
                dataset.transform = expert_transform
                for i in range(len(dataset.targets)):
                    if (dataset.targets[i] == 1 or dataset.targets[i] == 0 or dataset.targets[i] == 9 or dataset.targets[i] == 8):
                        indices.append (i)
                    
                target_set_1 = torch.utils.data.Subset(dataset, indices)
                loader_1 = torch.utils.data.DataLoader(dataset=target_set_1, 
                    batch_size=batch_size,shuffle=shuffle, sampler=sampler, **loader_kwargs)
                data.append(loader_1)
            elif (j==1):
                dataset.transform = expert_transform
                for i in range(len(dataset.targets)):
                    if (dataset.targets[i]== 8 or dataset.targets[i] == 0 or dataset.targets[i] == 5 or dataset.targets[i] == 2):
                        indices.append (i)

                target_set_2 = torch.utils.data.Subset(dataset, indices)
                loader_2 = torch.utils.data.DataLoader(dataset=target_set_2, 
                    batch_size=batch_size,shuffle=shuffle, sampler=sampler, **loader_kwargs)
                data.append(loader_2)
            elif (j==2):
                dataset.transform = expert_transform
                for i in range(len(dataset.targets)):
                    if (dataset.targets[i]== 2 or dataset.targets[i] == 5 or dataset.targets[i] == 4 or dataset.targets[i] == 6):
                        indices.append (i) 

                target_set_3 = torch.utils.data.Subset(dataset, indices)
                loader_3 = torch.utils.data.DataLoader(dataset=target_set_3, 
                    batch_size=batch_size,shuffle=shuffle, sampler=sampler, **loader_kwargs)
                data.append(loader_3)
            elif (j==3):
                dataset.transform = expert_transform
                for i in range(len(dataset.targets)):
                    if (dataset.targets[i]== 6 or dataset.targets[i] == 4 or dataset.targets[i] == 2 or dataset.targets[i] == 3 or dataset.targets[i] == 5):
                        indices.append (i)
                    
                target_set_4 = torch.utils.data.Subset(dataset, indices)
                loader_4 = torch.utils.data.DataLoader(dataset=target_set_4, 
                    batch_size=batch_size,shuffle=shuffle, sampler=sampler, **loader_kwargs)
                data.append(loader_4)
            elif (j==4):
                dataset.transform = expert_transform
                for i in range(len(dataset.targets)):
                    if (dataset.targets[i]== 7 or dataset.targets[i] == 2 or dataset.targets[i] == 4 or dataset.targets[i] == 3):
                        indices.append (i)
                
                target_set_5 = torch.utils.data.Subset(dataset, indices)
                loader_5 = torch.utils.data.DataLoader(dataset=target_set_5, 
                    batch_size=batch_size,shuffle=shuffle, sampler=sampler, **loader_kwargs)
                data.append(loader_5)
            else:
                for i in range(len(dataset.targets)):
                    dataset.transform = expert_transform
                    if (dataset.targets[i]== 7 or dataset.targets[i] == 0 or dataset.targets[i] == 3 or dataset.targets[i] == 8):
                        indices.append (i)
                target_set_6 = torch.utils.data.Subset(dataset, indices)
                loader_6 = torch.utils.data.DataLoader(dataset=target_set_6, 
                    batch_size=batch_size,shuffle=shuffle, sampler=sampler, **loader_kwargs)
                data.append(loader_6)

        return data
   
    else:
        print ("invalid partition mode!")

    # loader = torch.utils.data.DataLoader(dataset=target_set, batch_size=batch_size,
    #         shuffle=shuffle, sampler=sampler, num_workers=0, pin_memory=True)

def load_all_train_data(batch_size=64,sampler=None):
    cuda = True
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    
    # transform = tvt.Compose([
    #     tvt.ToTensor(),
    #     tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    transform = tvt.Compose([
        tvt.RandomCrop(32, padding=4),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if sampler is None:
        shuffle = True
    else:
        shuffle = False

    dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True,
            download=False, transform = transform)

    # https://ravimashru.dev/blog/2021-09-26-pytorch-subset/
    # Choose photos classed as '0'
    # idx0 = torch.tensor(dataset.targets) == 0
    # train_indices_0 = idx0.nonzero().reshape(-1)
    # # Randomly choose indices
    # indices_0 = torch.randperm(len(train_indices_0))[:600]
    # # Create a Subset
    # train_subset_0 = Subset(dataset, indices_0)

    # # Choose photos classed as '1'
    # idx1 = torch.tensor(dataset.targets) == 1
    # train_indices_1 = idx1.nonzero().reshape(-1)
    # # Randomly choose indices
    # indices_1 = torch.randperm(len(train_indices_1))[:600]
    # # Create a Subset
    # train_subset_1 = Subset(dataset, indices_1)

    # idx2 = torch.tensor(dataset.targets) == 2
    # train_indices_2 = idx2.nonzero().reshape(-1)
    # indices_2 = torch.randperm(len(train_indices_2))[:600]
    # # Create a Subset
    # train_subset_2 = Subset(dataset, indices_2)

    # idx3 = torch.tensor(dataset.targets) == 3
    # train_indices_3 = idx3.nonzero().reshape(-1)
    # indices_3 = torch.randperm(len(train_indices_3))[:600]
    # # Create a Subset
    # train_subset_3 = Subset(dataset, indices_3)

    # idx4 = torch.tensor(dataset.targets) == 4
    # train_indices_4 = idx4.nonzero().reshape(-1)
    # indices_4 = torch.randperm(len(train_indices_4))[:600]
    # # Create a Subset
    # train_subset_4 = Subset(dataset, indices_4)

    # idx5 = torch.tensor(dataset.targets) == 5
    # train_indices_5 = idx5.nonzero().reshape(-1)
    # indices_5 = torch.randperm(len(train_indices_5))[:600]
    # # Create a Subset
    # train_subset_5 = Subset(dataset, indices_5)

    # idx6 = torch.tensor(dataset.targets) == 6
    # train_indices_6 = idx6.nonzero().reshape(-1)
    # indices_6 = torch.randperm(len(train_indices_6))[:600]
    # # Create a Subset
    # train_subset_6 = Subset(dataset, indices_6)

    # idx7 = torch.tensor(dataset.targets) == 7
    # train_indices_7 = idx7.nonzero().reshape(-1)
    # indices_7 = torch.randperm(len(train_indices_7))[:600]
    # # Create a Subset
    # train_subset_7 = Subset(dataset, indices_7)

    # idx8 = torch.tensor(dataset.targets) == 8
    # train_indices_8 = idx8.nonzero().reshape(-1)
    # indices_8 = torch.randperm(len(train_indices_8))[:600]
    # # Create a Subset
    # train_subset_8 = Subset(dataset, indices_8)

    # idx9 = torch.tensor(dataset.targets) == 9
    # train_indices_9 = idx9.nonzero().reshape(-1)
    # indices_9 = torch.randperm(len(train_indices_9))[:600]
    # # Create a Subset
    # train_subset_9 = Subset(dataset, indices_9)

    # final_dataset = torch.utils.data.ConcatDataset([train_subset_0,train_subset_1,train_subset_2,train_subset_3,train_subset_4,
    #         train_subset_5,train_subset_6,train_subset_7,train_subset_8,train_subset_9])

    # train_loader = torch.utils.data.DataLoader(final_dataset, batch_size=batch_size,
    #         shuffle=True, sampler=sampler, **loader_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    return train_loader

def load_test_data(batch_size=1000, sampler=None):
    cuda = True
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

    # transform =  tvt.Compose([
    #     tvt.ToTensor(),
    #     tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    transform = tvt.Compose([
        tvt.RandomCrop(32, padding=4),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=False,
            download=False, transform=transform)
    # loader = torch.utils.data.DataLoader( dataset, batch_size=batch_size,
    #         shuffle=False, sampler=sampler, num_workers=4, pin_memory=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=False, sampler=sampler, **loader_kwargs)

    return loader
