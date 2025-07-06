import torch

config = {
    "latent_dim": 100,
    "image_size": 28,
    "image_channels": 1,
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "save_interval": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
