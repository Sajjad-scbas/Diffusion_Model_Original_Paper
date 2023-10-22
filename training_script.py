import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 

from train.dataloader import load_dataloader
from train.train import train_loop
from models.model import linear_beta_schedule, SimpleUNet  

IMG_SIZE = 64
    

if __name__  == "__main__":
    name = 'Flowers102'
    
    T = 300
    batch_size = 8 
    nb_epochs = 100
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    alphas, alphas_cumprod = linear_beta_schedule(timesteps=T)

    dataloader = load_dataloader(name, batch_size)
    
    model = SimpleUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(dataloader, model, criterion, optimizer, alphas_cumprod, T, epoch, device)
