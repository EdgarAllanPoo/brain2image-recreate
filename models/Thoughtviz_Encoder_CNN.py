import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Encoder_CNN(nn.Module):
    def __init__(self, channels, num_classes):
        super(Encoder_CNN, self).__init__()
        self.encoder = nn.Sequential(
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
            nn.Flatten(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, num_classes),
        )
    
    def forward(self, x):
        return self.encoder(x)
