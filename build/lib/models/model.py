import numpy as np
from functools import cache

import torch 
import torch.nn as nn 
import torch.nn.functional as F


######### Forward Pass - Noise scheduler #########
@cache
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    betas = torch.linspace(start, end, timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas, alphas_cumprod

def forward_diffusion_simple(x_0, t, alpha_cumprod, device='cpu'):
    batch_size = len(t)
    noise = torch.randn_like(x_0)
    return torch.sqrt(alpha_cumprod[t]).view(batch_size, 1, 1, 1).to(device) * x_0.to(device) + \
            torch.sqrt(1 - alpha_cumprod[t]).view(batch_size, 1, 1, 1).to(device) * noise.to(device), \
            noise.to(device)


######### Backward Pass - UNet #########
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim 
        
    def forward(self, t):
        device = t.device
        n = 10000
        half_dim = self.dim // 2
        emb =   np.log(n) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb 
        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super(ConvBlock, self).__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up :
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else :
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_n1 = nn.BatchNorm2d(out_channels)
        self.batch_n2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        h = self.batch_n1(self.relu(self.conv1(x)))
        # Time Embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.batch_n2(self.relu(self.conv2(h)))
        # UpSample or DownSample
        return self.transform(h)
    
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        num_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial Projection
        self.conv0 = nn.Conv2d(num_channels, down_channels[0], kernel_size=3, padding=1)
        
        # Down Sampling 
        self.downs = nn.ModuleList([
            ConvBlock(down_channels[i], down_channels[i+1], time_emb_dim) 
            for i in range(len(down_channels)-1)
        ])
        
        # Up Sampling 
        self.ups = nn.ModuleList([
            ConvBlock(up_channels[i], up_channels[i+1], time_emb_dim, up=True) 
            for i in range(len(up_channels)-1)
        ])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)
        
    def forward(self, x, timesteps):
        # Time Embedding
        t = self.time_mlp(timesteps)
        # Initial conv 
        x = self.conv0(x)
        # UNet
        residuals = []
        for down in self.downs :
            x = down(x, t)
            residuals.append(x)
        
        for up in self.ups :
            residual_x = residuals.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
            
        return self.output(x)


######### Simpling #########
@torch.no_grad()
def sample_timestep(model, x, t, timesteps): 
    alphas, alphas_cumprod = linear_beta_schedule(timesteps)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = (1-alphas_cumprod_prev)/(1-alphas_cumprod)*(1-alphas)
    
    model_mean = torch.sqrt(1./alphas[t])*(x - (1-alphas[t])/(torch.sqrt(1-alphas_cumprod[t])*model(x,t)))
    
    if t == 0 :
        return model_mean
    
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance[t])*noise
        






if __name__ == "__main__":
    # Number of of timesteps
    T = 300
    alphas_cumprod = linear_beta_schedule(timesteps=T)
    model = SimpleUNet()
    print('Done!')










