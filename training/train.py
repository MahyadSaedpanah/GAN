import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from training.utils import save_generated_images, save_image_grid, calculate_fid_is
from models.generator import Generator
from models.discriminator import Discriminator
from config import config

def train():
    device = config["device"]
    os.makedirs("outputs", exist_ok=True)

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

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

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
            real_loss = criterion(discriminator(real_images), valid)
            fake_loss = criterion(discriminator(gen_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch}/{config['epochs']}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if epoch in [25, 50, 100]:
            with torch.no_grad():
                z = torch.randn(64, config["latent_dim"], device=device)
                gen_samples = generator(z)
                save_generated_images(gen_samples, epoch)

                z = torch.randn(100, config["latent_dim"], device=device)
                fake_images = generator(z)
                save_image_grid(fake_images, folder=f"outputs/fake/epoch_{epoch}", prefix="fake")

                torch.save(generator.state_dict(), f"outputs/generator_epoch_{epoch}.pth")
                torch.save(discriminator.state_dict(), f"outputs/discriminator_epoch_{epoch}.pth")

    # Final loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss", color="blue")
    plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss", color="red")
    for mark in [25, 50, 100]:
        plt.axvline(x=mark, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Overall Loss During Training")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/loss_plot_overall.png")
    plt.close()

    # Loss report
    with open("outputs/loss_report.txt", "w") as f:
        f.write("Loss Averages per Epoch Range\n===============================\n\n")
        for (start, end) in [(0, 25), (25, 50), (50, 100)]:
            avg_g = sum(g_losses[start:end]) / (end - start)
            avg_d = sum(d_losses[start:end]) / (end - start)
            f.write(f"Epochs {start+1}-{end}:\n")
            f.write(f"    Generator Loss: {avg_g:.4f}\n")
            f.write(f"    Discriminator Loss: {avg_d:.4f}\n\n")

    # FID/IS Evaluation
    with open("outputs/fid_is_report.txt", "w") as f:
        for epoch in [25, 50, 100]:
            fake_dir = f"outputs/fake/epoch_{epoch}"
            fid, is_score = calculate_fid_is(real_dir="outputs/real", fake_dir=fake_dir, device=device)
            f.write(f"Epoch {epoch}: FID = {fid:.2f}, IS = {is_score:.2f}\n")
            print(f"[Eval] Epoch {epoch} => FID: {fid:.2f}, IS: {is_score:.2f}")

if __name__ == "__main__":
    train()
