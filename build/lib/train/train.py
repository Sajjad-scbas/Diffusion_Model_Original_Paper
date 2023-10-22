import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from einops import rearrange

from models.model import sample_timestep, forward_diffusion_simple

from tqdm import tqdm

IMG_SIZE = 64

def get_loss(model, x_0, t, alphas_cumprod, criterion = nn.L1Loss()):
    x_noisy, noise = forward_diffusion_simple(x_0, t, alphas_cumprod)
    noise_pred = model(x_noisy, t)
    return criterion(noise, noise_pred)


######### Plotting images durig the training #########
@torch.no_grad()
def sample_plot_image(model, T, device = 'cpu'):
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device = device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    
    num_images = 10
    stepsize = T // num_images
    
    for idx, t in enumerate(range(0, T, stepsize)):
        t = torch.LongTensor([t], device=device) 
        img = sample_timestep(model, img, t, T)
        plt.subplot(1, num_images, idx+1)
        plt.imshow(rearrange(img, '1 c h w -> h w c').detach().cpu().numpy())
    plt.show()


def train_loop(dataloader, model, loss_fn, optimizer, alphas_cumprod, T, epoch, device):

    nb_batchs = dataloader.batch_size
    batch_size = dataloader.batch_size
    model.train()
    for idx, x in enumerate(tqdm(dataloader)):

        t = torch.randint(0, T, (batch_size,), device=device)
        loss = get_loss(model, x[0], t, alphas_cumprod, loss_fn)
        
        #Back Pass
        loss.backward()
                
        #Weights Update
        optimizer.step()
        optimizer.zero_grad()
        

        print(f'training-loss : {loss.item():>7f} | [{(idx+1)}/ {len(dataloader)}]')

        if epoch % 5 == 0 :
            sample_plot_image(model, T)
