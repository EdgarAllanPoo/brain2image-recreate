import torch
import torch.nn as nn
import torch.nn.init as init

class MoG_RGB(nn.Module):
    def __init__(self, noise_dim, features_dim):
        super(MoG_RGB, self).__init__()
        self.std = nn.Parameter(torch.empty(noise_dim))
        self.mean = nn.Parameter(torch.empty(noise_dim))
        self.initialize_parameters()

        self.hidden_eeg_features = nn.Linear(features_dim, noise_dim)

    def initialize_parameters(self):
        init.uniform_(self.std, a=-0.2, b=0.2)
        init.uniform_(self.mean, a=-1, b=1)

    def forward(self, noise, eeg_features):
        output = noise * self.std
        output += self.mean

        intermediate_features = self.hidden_eeg_features(eeg_features)

        output *= intermediate_features

        return output
