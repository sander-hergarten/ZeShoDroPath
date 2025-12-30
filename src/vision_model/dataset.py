"""Dataset loader for diffusion model training."""

import io
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms


class ImageDiffusionIterableDataset(IterableDataset):
    """Iterable dataset for streaming large parquet files that don't fit in memory."""

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
        parquet_path = str(Path(parquet_file).resolve())
        self.dataset = load_dataset(
            "parquet", data_files=parquet_path, split="train", streaming=True
        )
        self.image_size = image_size
        self.iteration_steps = iteration_steps

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
                    transforms.ToTensor(),
                ]
            )
        else:
            self.mask_transform = mask_transform

    def _process_sample(self, sample: dict) -> dict:
        """Process a single sample from the dataset."""
        # Load RGB image (image_0) - used as conditioning
        rgb_data = sample["image"]
        if isinstance(rgb_data, bytes):
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert("RGB")
        else:
            rgb_image = rgb_data.convert("RGB")

        # Load clean mask (image_N - least noisy, last mask column)
        clean_data = sample["clean"]
        if isinstance(clean_data, bytes):
            clean_mask = Image.open(io.BytesIO(clean_data)).convert("L")
        else:
            clean_mask = clean_data.convert("L")

        # Load corresponding noisy mask
        noisy_data = sample["noisy"]
        if isinstance(noisy_data, bytes):
            noisy_mask = Image.open(io.BytesIO(noisy_data)).convert("L")
        else:
            noisy_mask = noisy_data.convert("L")

        # Apply transforms
        rgb_tensor = self.rgb_transform(rgb_image)
        clean_mask_tensor = self.mask_transform(clean_mask)
        noisy_mask_tensor = self.mask_transform(noisy_mask)

        # Timestep: 0 = most noisy, 1 = clean
        timestep = sample["timestep"]

        return {
            "rgb": rgb_tensor,
            "noisy_mask": noisy_mask_tensor,
            "clean_mask": clean_mask_tensor,
            "timestep": torch.tensor(timestep, dtype=torch.float32),
        }

    def __iter__(self):
        """Iterate over the streaming dataset."""
        for sample in self.dataset:
            yield self._process_sample(sample)


class ImageDiffusionDataset(Dataset[dict[str, str]]):
    """Dataset that loads images from parquet and creates noisy-clean mask pairs.

    Column 0 (image_0): RGB image used as latent/conditioning
    Columns 1+ (image_1 to image_N): Progressively less noisy black and white masks

    Supports both in-memory and streaming modes for large datasets.
    """

    def __init__(
        self,
        parquet_file: str,
        image_size: int = 512,
        iteration_steps: int = 10,
        rgb_transform=None,
        mask_transform=None,
        streaming: bool = False,
    ):
        """
        Args:
            parquet_file: Path to the parquet file
            image_size: Target image size (default: 512)
            iteration_steps: Number of mask steps (default: 10, so columns 1-10)
            rgb_transform: Optional transform for RGB images
            mask_transform: Optional transform for mask images
            streaming: If True, use streaming mode (doesn't load entire dataset into memory).
                       Note: For streaming, use ImageDiffusionIterableDataset instead.
        """
        # Ensure parquet_file is a string and resolve to absolute path
        parquet_path = str(Path(parquet_file).resolve())
        self.streaming = streaming
        self.image_size = image_size
        self.iteration_steps = iteration_steps
        self.num_mask_levels = iteration_steps  # image_1 to image_N

        # Load dataset (non-streaming for regular Dataset)
        self.dataset = load_dataset(
            "parquet", data_files=parquet_path, split="train", streaming=False
        )
        self._dataset_length = len(self.dataset)

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
        return self._dataset_length

    def _process_sample(self, sample: dict) -> dict:
        """Process a single sample from the dataset."""
        # Load RGB image (image_0) - used as conditioning
        rgb_data = sample["image_0"]
        if isinstance(rgb_data, bytes):
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert("RGB")
        else:
            rgb_image = rgb_data.convert("RGB")

        # Load clean mask (image_N - least noisy, last mask column)
        clean_mask_idx = self.iteration_steps  # image_N
        clean_data = sample[f"image_{clean_mask_idx}"]
        if isinstance(clean_data, bytes):
            clean_mask = Image.open(io.BytesIO(clean_data)).convert("L")  # Grayscale
        else:
            clean_mask = clean_data.convert("L")

        # Randomly select a noise level (1 = most noisy, N = least noisy)
        # We'll use this to select which mask to use as noisy input
        # noise_level ranges from 1 to N-1 (excluding the clean mask at N)
        noise_level = np.random.randint(1, clean_mask_idx)  # 1 to N-1

        # Load corresponding noisy mask
        noisy_data = sample[f"image_{noise_level}"]
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

    def __getitem__(self, idx):
        """Returns (rgb_image, noisy_mask, clean_mask, timestep).

        Returns:
            dict with keys:
                - rgb: RGB image tensor (3, H, W) - conditioning/latent
                - noisy_mask: Noisy mask tensor (1, H, W)
                - clean_mask: Clean mask tensor (1, H, W)
                - timestep: Timestep tensor (scalar)
        """
        sample = self.dataset[idx]
        return self._process_sample(sample)
