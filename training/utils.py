
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def save_generated_images(samples, epoch, output_dir="outputs/samples"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"epoch_{epoch}.png")
    save_image(samples, file_path, nrow=8, normalize=True)

def save_image_grid(images, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        img = (img + 1) / 2
        img = transforms.ToPILImage()(img.cpu())
        img.save(os.path.join(folder, f"{prefix}_{i}.png"))

def save_real_images(num_samples=100, folder="outputs/real"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = MNIST(root="data", train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    real_imgs, _ = next(iter(loader))
    save_image_grid(real_imgs[:num_samples], folder, "real")

def calculate_fid_is(real_dir, fake_dir, device="cuda"):
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=(device == "cuda"),
        isc=True,
        fid=True,
        verbose=True
    )
    return metrics['frechet_inception_distance'], metrics['inception_score_mean']
