import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImageEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Input: 3x64x64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> 32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 128x8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 256x4x4
            nn.ReLU(),
            nn.Flatten(),  # -> 4096
            nn.Linear(4096, latent_dim),  # -> latent_dim
        )

    def forward(self, x):
        # x shape: (batch_size, 3, 64, 64)
        z = self.encoder(x)
        return z