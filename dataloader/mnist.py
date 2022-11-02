import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from os.path import join
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import random


def load_train_data(partition_mode,batch_size=128,sampler=None):
    cuda = True
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    
    train_set = MNIST(join('datasets', 'mnist'), train=True, download=False,
            transform=Compose([
                   Resize((28, 28)),
                   ToTensor(),
                   Normalize((0.1307,),(0.308,)),
                  ]))
    
    # The data is intelligently partitioned
    data = []
    if (partition_mode=="intelligent"):
        for i in range (0,4):
            indices = []
            if (i==0):
                for j in range(0,len(train_set.targets)):
                    if (train_set.targets[j]== 1 or train_set.targets[j] == 4 or train_set.targets[j] == 9 or train_set.targets[j] == 7):
                        indices.append (j)
                target_set_1 = torch.utils.data.Subset(train_set, indices)
                loader_1 = torch.utils.data.DataLoader(dataset=target_set_1, 
                    batch_size=batch_size,**loader_kwargs)
                data.append (loader_1)

            elif (i==1):
                for j in range(0,len(train_set.targets)):
                    if (train_set.targets[j]== 5 or train_set.targets[j] == 8 or train_set.targets[j] == 3 or train_set.targets[j] == 2):
                        indices.append (j)
                target_set_2 = torch.utils.data.Subset(train_set, indices)
                loader_2 = torch.utils.data.DataLoader(dataset=target_set_2, 
                    batch_size=batch_size,**loader_kwargs)
                data.append (loader_2)

            elif (i==2):
                for j in range(0,len(train_set.targets)):
                    if (train_set.targets[j]== 0 or train_set.targets[j] == 6 or train_set.targets[j] == 5 or train_set.targets[j] == 3):
                        indices.append (j)
                target_set_3 = torch.utils.data.Subset(train_set, indices)
                loader_3 = torch.utils.data.DataLoader(dataset=target_set_3, 
                    batch_size=batch_size,**loader_kwargs)
                data.append (loader_3)

            else:
                for j in range(0,len(train_set.targets)):
                    if (train_set.targets[j]== 1 or train_set.targets[j] == 2 or train_set.targets[j] == 8 or train_set.targets[j] == 8):
                        indices.append (j)
                target_set_4 = torch.utils.data.Subset(train_set, indices)
                loader_4 = torch.utils.data.DataLoader(dataset=target_set_4, 
                    batch_size=batch_size,**loader_kwargs)
                data.append (loader_4)
            
        return data

    else:
        print ("invalid partition mode!")
    # elif (num==4):
    #     indices = (train_set.targets == 0) | (train_set.targets == 6) | (train_set.targets == 5) | (train_set.targets == 3)
    # else:
    #     indices = (train_set.targets == 2) | (train_set.targets == 1) | (train_set.targets == 8) | (train_set.targets == 7)
    
    # print (train_set.data, train_set.targets)
    # a = range(0,60000)
    # if (trainset_num==10):
    #     trainset = Subset(train_set, list(i for i in a if (i%10==0)))
    # else:
    #     trainset = Subset(train_set, list(i for i in a if (i%12==0)))

    # return samples 

    # train_loader = DataLoader(train_set,
    #     batch_size=batch_size, shuffle=True, **loader_kwargs)

    # return train_loader

def load_all_train_data(batch_size=128,sampler=None):
    cuda = True
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    
    train_set = MNIST(join('datasets', 'mnist'), train=True, download=True,
            transform=Compose([
                   Resize((28, 28)),
                   ToTensor(),
                   Normalize((0.1307,),(0.308,)),
                  ]))

    train_loader = DataLoader(train_set,
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    return train_loader

def load_test_data(batch_size=1000, sampler=None):
    
    cuda = True
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    
    test_loader = DataLoader(
        MNIST(join('datasets', 'mnist'), train=False, download=True,
            transform=Compose([
                   Resize((28, 28)),
                   ToTensor(),
                   Normalize((0.1307,),(0.308,)),
                    ])),
        batch_size= batch_size, shuffle=False,sampler=sampler, **loader_kwargs)

    print ("len")
    print (len(test_loader.sampler))
    return test_loader


