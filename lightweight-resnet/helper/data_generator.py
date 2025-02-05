import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, Normalize, Resize, RandomHorizontalFlip

import pickle

def unpickle(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

cifar100_train_transform = Compose([
    Resize(224, antialias=True),
    Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    RandomHorizontalFlip(),
    ])

cifar100_test_transform = Compose([
    Resize(224, antialias=True),
    Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ])

def cifar100(path, batch_size=256, train : bool = True):
    data_dict = unpickle(path) # ./datasets/train or ./datasets/test
    images = torch.tensor( data_dict[b'data'] ).view(-1, 3, 32, 32).to(torch.float32)
    if train:
        images = cifar100_train_transform(images / 255.0)
    else:
        images = cifar100_test_transform(images / 255.0)
    labels = torch.tensor(data_dict[b'fine_labels'])
    
    data_loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size)
    return data_loader