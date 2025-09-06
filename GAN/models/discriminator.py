import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),     # (28x28) → (14x14)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),   # (14x14) → (7x7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # (7x7) → (3x3)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),                 # → 256*3*3
            nn.Linear(256 * 3 * 3, 1024), # Fully Connected Layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),           # Output Layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
