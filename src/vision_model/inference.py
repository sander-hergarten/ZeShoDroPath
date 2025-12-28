"""Inference script for the Bernoulli Diffusion Model."""

import io
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

from .dataset import ImageDiffusionDataset
from .model import UNet


def load_model(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    iteration_steps: int = 10,
):
    """Load trained model from checkpoint."""
    model = UNet(
        rgb_channels=3,
        mask_channels=1,
        out_channels=1,
        base_channels=64,
        time_emb_dim=128,
        channel_multipliers=(1, 2, 4, 8),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs, loss: {checkpoint['loss']:.4f}")

    return model


def denoise_mask(
    model: torch.nn.Module,
    rgb: torch.Tensor,
    noisy_mask: torch.Tensor,
    timestep: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Denoise a mask given RGB conditioning."""
    model.eval()
    with torch.no_grad():
        rgb = rgb.unsqueeze(0).to(device)  # (1, 3, H, W)
        noisy_mask = noisy_mask.unsqueeze(0).to(device)  # (1, 1, H, W)
        timestep_tensor = torch.tensor([timestep], dtype=torch.float32).to(device)

        pred_clean_mask = model(rgb, noisy_mask, timestep_tensor)
        pred_clean_mask = torch.clamp(pred_clean_mask, 0, 1)

        return pred_clean_mask.squeeze(0).cpu()  # (1, H, W)


def inference_from_dataset(
    checkpoint_path: str,
    parquet_file: str,
    num_samples: int = 5,
    image_size: int = 512,
    output_dir: str = "./inference_outputs",
    iteration_steps: int = 10,
):
    """Run inference on samples from the dataset."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    model = load_model(checkpoint_path, device, iteration_steps)

    # Load dataset
    dataset = ImageDiffusionDataset(
        parquet_file, image_size=image_size, iteration_steps=iteration_steps
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Sample random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    print(f"\nRunning inference on {num_samples} samples...")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        rgb = sample["rgb"]  # (3, H, W)
        noisy_mask = sample["noisy_mask"]  # (1, H, W)
        clean_mask = sample["clean_mask"]  # (1, H, W)
        timestep = sample["timestep"].item()

        # Denoise
        pred_clean_mask = denoise_mask(model, rgb, noisy_mask, timestep, device)

        # Convert to numpy for visualization
        rgb_np = rgb.permute(1, 2, 0).numpy()  # (H, W, 3)
        noisy_mask_np = noisy_mask.squeeze(0).numpy()  # (H, W)
        clean_mask_np = clean_mask.squeeze(0).numpy()  # (H, W)
        pred_mask_np = pred_clean_mask.squeeze(0).numpy()  # (H, W)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # RGB conditioning image
        axes[0, 0].imshow(rgb_np)
        axes[0, 0].set_title("RGB Conditioning Image")
        axes[0, 0].axis("off")

        # Noisy mask
        axes[0, 1].imshow(noisy_mask_np, cmap="gray")
        axes[0, 1].set_title(f"Noisy Mask (t={timestep:.2f})")
        axes[0, 1].axis("off")

        # Predicted clean mask
        axes[1, 0].imshow(pred_mask_np, cmap="gray")
        axes[1, 0].set_title("Predicted Clean Mask")
        axes[1, 0].axis("off")

        # Ground truth clean mask
        axes[1, 1].imshow(clean_mask_np, cmap="gray")
        axes[1, 1].set_title("Ground Truth Clean Mask")
        axes[1, 1].axis("off")

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"inference_{i+1}_idx_{idx}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved inference result {i+1} to {output_path}")

    print(f"\nInference complete! Results saved to {output_dir}")


def inference_from_files(
    checkpoint_path: str,
    rgb_path: str,
    mask_path: str,
    timestep: float = 0.0,
    output_path: str = "./denoised_mask.png",
    image_size: int = 512,
    iteration_steps: int = 10,
):
    """Denoise a mask file given RGB and mask image files."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    model = load_model(checkpoint_path, device, iteration_steps)

    # Load and preprocess images
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    rgb_image = Image.open(rgb_path).convert("RGB")
    rgb_tensor = transform(rgb_image)  # (3, H, W)

    mask_image = Image.open(mask_path).convert("L")  # Grayscale
    mask_tensor = transform(mask_image)  # (1, H, W)

    # Denoise
    print(f"Denoising mask with timestep {timestep}...")
    pred_clean_mask = denoise_mask(model, rgb_tensor, mask_tensor, timestep, device)

    # Convert to PIL and save
    pred_np = pred_clean_mask.squeeze(0).numpy()  # (H, W)
    pred_np = (pred_np * 255).astype(np.uint8)
    pred_image = Image.fromarray(pred_np, mode="L")

    pred_image.save(output_path)
    print(f"Denoised mask saved to {output_path}")


@dataclass
class DatasetInferenceArgs:
    """Arguments for dataset inference."""

    checkpoint: str
    """Path to model checkpoint"""
    parquet_file: str = "dataset.parquet"
    """Path to parquet file"""
    num_samples: int = 5
    """Number of samples to process"""
    image_size: int = 512
    """Image size"""
    output_dir: str = "./inference_outputs"
    """Output directory"""
    iteration_steps: int = 10
    """Number of mask iteration steps"""


@dataclass
class FilesInferenceArgs:
    """Arguments for file-based inference."""

    checkpoint: str
    """Path to model checkpoint"""
    rgb_path: str
    """Path to RGB conditioning image"""
    mask_path: str
    """Path to noisy mask image"""
    timestep: float = 0.0
    """Timestep (0.0 = most noisy, 1.0 = clean)"""
    output_path: str = "./denoised_mask.png"
    """Output mask path"""
    image_size: int = 512
    """Image size"""
    iteration_steps: int = 10
    """Number of mask iteration steps"""


def main_dataset(args: DatasetInferenceArgs):
    """Inference on dataset samples."""
    inference_from_dataset(
        checkpoint_path=args.checkpoint,
        parquet_file=args.parquet_file,
        num_samples=args.num_samples,
        image_size=args.image_size,
        output_dir=args.output_dir,
        iteration_steps=args.iteration_steps,
    )


def main_files(args: FilesInferenceArgs):
    """Inference on RGB and mask image files."""
    inference_from_files(
        checkpoint_path=args.checkpoint,
        rgb_path=args.rgb_path,
        mask_path=args.mask_path,
        timestep=args.timestep,
        output_path=args.output_path,
        image_size=args.image_size,
        iteration_steps=args.iteration_steps,
    )


if __name__ == "__main__":
    tyro.cli(
        {
            "dataset": main_dataset,
            "files": main_files,
        }
    )

