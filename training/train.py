import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from training.utils import save_generated_images
from models.generator import Generator
from models.discriminator import Discriminator
from config import config


def train():
    device = config["device"]
    
    # ✅ داده‌ها
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # برای تطبیق با Tanh
    ])
    dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # ✅ مدل‌ها
    generator = Generator(config["latent_dim"]).to(device)
    discriminator = Discriminator().to(device)

    # ✅ توابع خطا و بهینه‌ساز
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))

    # ✅ حلقه آموزش
    for epoch in range(1, config["epochs"] + 1):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # برچسب‌های واقعی و جعلی
            valid = torch.ones(batch_size, 1, device=device)
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
            real_loss = criterion(discriminator(real_images), valid)
            fake_loss = criterion(discriminator(gen_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        # ✅ نمایش خروجی
        print(f"[Epoch {epoch}/{config['epochs']}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
        if epoch % config["save_interval"] == 0 or epoch == 1:
            with torch.no_grad():
                z = torch.randn(64, config["latent_dim"], device=device)
                gen_samples = generator(z)
                save_generated_images(gen_samples, epoch)

