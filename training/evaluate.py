
from training.utils import calculate_fid_is, save_real_images
from config import config

if __name__ == "__main__":
    save_real_images()

    fid, is_score = calculate_fid_is(
        real_dir="outputs/real",
        fake_dir="outputs/fake/epoch_100_large",
        device=config["device"]
    )

    print(f"FID: {fid:.2f}, Inception Score: {is_score:.2f}")
