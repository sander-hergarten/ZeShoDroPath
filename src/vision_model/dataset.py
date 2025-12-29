"""Dataset loader for diffusion model training."""

import io
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDiffusionDataset(Dataset):
    """Dataset that loads images from parquet and creates noisy-clean mask pairs.
    
    Column 0 (image_0): RGB image used as latent/conditioning
    Columns 1+ (image_1 to image_N): Progressively less noisy black and white masks
    """

    def __init__(
        self,
        parquet_file: str,
        image_size: int = 512,
        iteration_steps: int = 10,
        rgb_transform=None,
        mask_transform=None,
    ):
        """
        Args:
            parquet_file: Path to the parquet file
            image_size: Target image size (default: 512)
            iteration_steps: Number of mask steps (default: 10, so columns 1-10)
            rgb_transform: Optional transform for RGB images
            mask_transform: Optional transform for mask images
        """
        # Ensure parquet_file is a string and resolve to absolute path
        parquet_path = str(Path(parquet_file).resolve())
        self.dataset = load_dataset("parquet", data_files=parquet_path, split="train")
        self.image_size = image_size
        self.iteration_steps = iteration_steps
        self.num_mask_levels = iteration_steps  # image_1 to image_N

        # Transform for RGB images (image_0)
        if rgb_transform is None:
            self.rgb_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.rgb_transform = rgb_transform

        # Transform for mask images (image_1 to image_N) - grayscale
        if mask_transform is None:
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),  # Converts to [0, 1] range
                ]
            )
        else:
            self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns (rgb_image, noisy_mask, clean_mask, timestep).
        
        Returns:
            dict with keys:
                - rgb: RGB image tensor (3, H, W) - conditioning/latent
                - noisy_mask: Noisy mask tensor (1, H, W)
                - clean_mask: Clean mask tensor (1, H, W)
                - timestep: Timestep tensor (scalar)
        """
        # Load RGB image (image_0) - used as conditioning
        rgb_data = self.dataset[idx]["image_0"]
        if isinstance(rgb_data, bytes):
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert("RGB")
        else:
            rgb_image = rgb_data.convert("RGB")

        # Load clean mask (image_N - least noisy, last mask column)
        clean_mask_idx = self.iteration_steps  # image_N
        clean_data = self.dataset[idx][f"image_{clean_mask_idx}"]
        if isinstance(clean_data, bytes):
            clean_mask = Image.open(io.BytesIO(clean_data)).convert("L")  # Grayscale
        else:
            clean_mask = clean_data.convert("L")

        # Randomly select a noise level (1 = most noisy, N = least noisy)
        # We'll use this to select which mask to use as noisy input
        # noise_level ranges from 1 to N-1 (excluding the clean mask at N)
        noise_level = np.random.randint(1, clean_mask_idx)  # 1 to N-1

        # Load corresponding noisy mask
        noisy_data = self.dataset[idx][f"image_{noise_level}"]
        if isinstance(noisy_data, bytes):
            noisy_mask = Image.open(io.BytesIO(noisy_data)).convert("L")  # Grayscale
        else:
            noisy_mask = noisy_data.convert("L")

        # Apply transforms
        rgb_tensor = self.rgb_transform(rgb_image)  # (3, H, W)
        clean_mask_tensor = self.mask_transform(clean_mask)  # (1, H, W)
        noisy_mask_tensor = self.mask_transform(noisy_mask)  # (1, H, W)

        # Timestep: 0 = most noisy, 1 = clean
        # Map noise_level (1 to N-1) to timestep (0.0 to ~0.9)
        # Note: noise_level 1 is most noisy, clean_mask_idx is least noisy
        timestep = (noise_level - 1) / (clean_mask_idx - 1)

        return {
            "rgb": rgb_tensor,
            "noisy_mask": noisy_mask_tensor,
            "clean_mask": clean_mask_tensor,
            "timestep": torch.tensor(timestep, dtype=torch.float32),
        }

