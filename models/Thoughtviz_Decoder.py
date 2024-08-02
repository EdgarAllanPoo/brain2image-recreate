import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            
        )