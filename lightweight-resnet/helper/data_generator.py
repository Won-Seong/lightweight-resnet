import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, Normalize

import pickle

def unpickle(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

cifar100_transform = Compose([
    Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ])

def cifar100(path, batch_size=256):
    data_dict = unpickle(path) # ./datasets/train or ./datasets/test
    images = torch.tensor( data_dict[b'data'] ).view(-1, 3, 32, 32).to(torch.float32)
    images = cifar100_transform(images / 255.0)
    labels = torch.tensor(data_dict[b'fine_labels'])
    
    data_loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size)
    return data_loader