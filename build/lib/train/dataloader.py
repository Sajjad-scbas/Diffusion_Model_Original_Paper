import numpy as np

import torch 
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data.data import get_data


IMG_SIZE = 64

def load_dataloader(name, batch_size):
    data_transformers = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(lambda t: (t*2)-1)
    ])
    data = get_data(name, data_transformers, root='./data/')
    dataloader = DataLoader(data, batch_size, shuffle=True)
    return dataloader     




