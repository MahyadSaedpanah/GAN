
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from training.utils import save_generated_images, save_image_grid
from models.generator import Generator
from models.discriminator import Discriminator
from config import config

def train():
    device = config["device"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    generator = Generator(config["latent_dim"]).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))

    g_losses = []
    d_losses = []

    for epoch in range(1, config["epochs"] + 1):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Label smoothing for real labels
            valid = torch.full((batch_size, 1), 0.9, device=device)  # real label = 0.9
            fake = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Train Generator
            # ---------------------
            z = torch.randn(batch_size, config["latent_dim"], device=device)
            gen_images = generator(z)

            g_loss = criterion(discriminator(gen_images), valid)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Inject Gaussian noise into inputs
            noisy_real = real_images + 0.05 * torch.randn_like(real_images)
            noisy_fake = gen_images.detach() + 0.05 * torch.randn_like(gen_images.detach())

            real_loss = criterion(discriminator(noisy_real), valid)
            fake_loss = criterion(discriminator(noisy_fake), fake)
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch}/{config['epochs']}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if epoch in [1, 25, 50, 100]:
            with torch.no_grad():
                z = torch.randn(64, config["latent_dim"], device=device)
                gen_samples = generator(z)
                save_generated_images(gen_samples, epoch)

                z = torch.randn(100, config["latent_dim"], device=device)
                fake_images = generator(z)
                save_image_grid(fake_images, folder=f"outputs/fake/epoch_{epoch}", prefix="fake")

            torch.save(generator.state_dict(), f"outputs/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"outputs/discriminator_epoch_{epoch}.pth")

            plt.figure(figsize=(10, 5))
            plt.plot(g_losses, label="Generator Loss")
            plt.plot(d_losses, label="Discriminator Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Loss During Training")
            plt.savefig(f"outputs/loss_plot_epoch_{epoch}.png")
            plt.close()
