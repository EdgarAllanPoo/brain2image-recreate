import torch
import torch.nn as nn

from models.custom_layers.ThoughtViz_MoG import MoG

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.gaussian_layer = MoG(noise_dim)
        self.gen_p1 = nn.Sequential(
            nn.Linear(200, 1024),
            nn.Tanh(),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, 128 * 7 * 7),
            nn.Tanh(),
        )
        self.gen_p2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, eeg_features):
        output = self.gaussian_layer(noise)
        output = torch.cat((noise, eeg_features), dim=1)
        output = self.gen_p1(output)
        output = torch.reshape(output, (output.shape[0], 128, 7, 7))
        output = self.gen_p2(output)

        return output
    
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_dim, 64, kernel_size=(5, 5), padding=2),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(5,5)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(3200, 1024),
            nn.Tanh(),           
        )
        self.fake = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        output = self.disc(img)
        output = self.fake(output)

        return output

# test
# batch_size = 32
# noise_dim = (batch_size, 100) 
# eeg_dim = (batch_size, 100)    

# noise = torch.randn(noise_dim)
# eeg_features = torch.randn(eeg_dim)

# generator = Generator(noise_dim)

# output = generator.forward(noise, eeg_features)
# print("Generator output shape:", output.shape)

# img_dim = 1
# discriminator = Discriminator(img_dim)

# output = discriminator(output)
# print("Discriminator output shape:", output.shape)
# print(output)
# print(output[:5])