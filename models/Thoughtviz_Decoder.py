import torch
import torch.nn as nn

from custom_layers.ThoughtViz_MoG import MoG

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
        # Mengubah ukuran menjadi (BATCH_SIZE, 1, NOISE_DIM)
        output = self.gaussian_layer(noise)
        output = torch.cat((noise, eeg_features), dim=1)
        output = self.gen_p1(output)
        output = torch.reshape(output, (output.shape[0], 128, 7, 7))
        output = self.gen_p2(output)

        return output
    
# class Discriminator(nn.Module):
#     def __init__(self, img_shape, output_dim):
#         super(Discriminator, self).__init__()


# Parameter untuk pengujian
batch_size = 32
noise_dim = (batch_size, 100)  # 100 adalah dimensi dari noise
eeg_dim = (batch_size, 100)    # 100 adalah dimensi dari fitur EEG

# Buat input buatan
noise = torch.randn(noise_dim)
eeg_features = torch.randn(eeg_dim)

# Inisialisasi model
generator = Generator(noise_dim)

# Forward pass
output = generator.forward(noise, eeg_features)

# Output hasil
print("Output shape:", output.shape)