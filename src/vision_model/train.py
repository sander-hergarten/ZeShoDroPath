"""Training script for Bernoulli diffusion model."""

import os

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ImageDiffusionDataset, ImageDiffusionIterableDataset
from .model import UNet
from .trainer import BernoulliDiffusionModel


def train(
    parquet_file: str,
    output_dir: str = "./checkpoints",
    batch_size: int = 4,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    image_size: int = 512,
    num_workers: int = 4,
    save_every: int = 10,
    iteration_steps: int = 10,
    streaming: bool = True,
):
    """Train the diffusion model.

    Args:
        streaming: If True, use streaming mode for datasets that don't fit in memory.
                   When streaming=True, shuffle is disabled and dataset length is unknown.
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Dataset and dataloader
    print("Loading dataset...")
    if streaming:
        print("Using streaming mode (dataset won't be loaded into memory)")
        dataset = ImageDiffusionIterableDataset(
            parquet_file, image_size=image_size, iteration_steps=iteration_steps
        )
        # For IterableDataset, shuffle must be False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # IterableDataset doesn't support shuffle
            num_workers=num_workers,
            pin_memory=device != "cpu",
        )
        print("Dataset: Streaming (size unknown)")
    else:
        dataset = ImageDiffusionDataset(
            parquet_file, image_size=image_size, iteration_steps=iteration_steps
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device != "cpu",
        )
        print(f"Dataset size: {len(dataset)}")

    print(f"Batch size: {batch_size}")
    if not streaming:
        print(f"Number of batches per epoch: {len(dataloader)}")
    print(f"Iteration steps: {iteration_steps}")

    # Model - RGB (3) + mask (1) = 4 input channels, 1 output channel
    model = UNet(
        rgb_channels=3,
        mask_channels=1,
        out_channels=1,
        base_channels=64,
        time_emb_dim=128,
        channel_multipliers=(1, 2, 4, 8),
    )

    trainer = BernoulliDiffusionModel(
        model=model,
        device=device,
        learning_rate=learning_rate,
    )

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            result = trainer.train_step(batch)
            epoch_losses.append(result["loss"])
            pbar.set_postfix({"loss": f"{result['loss']:.4f}"})

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(
                output_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1, avg_loss)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training complete!")


if __name__ == "__main__":
    tyro.cli(train)
