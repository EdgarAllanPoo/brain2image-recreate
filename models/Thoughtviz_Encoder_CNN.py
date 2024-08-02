import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pickle

class Encoder_CNN(nn.Module):
    def __init__(self, channels, num_classes):
        super(Encoder_CNN, self).__init__()
        self.conv2d = nn.Sequential(
            nn.BatchNorm2d(14),
            nn.Conv2d(14, 32, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 25, kernel_size=(channels, 1)),
            nn.ReLU(),
            nn.Conv2d(25, 50, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1)),
            nn.Conv2d(50, 100, kernel_size=(4, 1)),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.ffnn = nn.Sequential(
            nn.BatchNorm1d(100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, num_classes),
        )
    
    def forward(self, x):
        out = self.conv2d(x)
        out = features = self.flatten(out)
        out = self.ffnn(out)
        return out, features