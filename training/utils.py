import os
import torch
import torchvision
from torchvision.utils import save_image

def save_generated_images(samples, epoch, output_dir="outputs/samples"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"epoch_{epoch}.png")
    save_image(samples, file_path, nrow=8, normalize=True)
