import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),     # ظرفیت بیشتر
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),

            nn.Unflatten(1, (256, 7, 7)),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 3, 1, 1),     # Output
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
