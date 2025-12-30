"""Main entry point for vision_model module."""

from pathlib import Path
import tyro

from . import train as p_train
from . import inference

app = tyro.extras.SubcommandApp()


@app.command
def train(
    parquet_file: Path = Path("dataset.parquet"),
    output_dir: Path = Path("./checkpoints"),
    batch_size: int = 4,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    image_size: int = 512,
    num_workers: int = 4,
    save_every: int = 10,
    iteration_steps: int = 10,
    streaming: bool = True,
) -> None:
    """Train the diffusion model.
    
    Args:
        streaming: Use streaming mode for large datasets that don't fit in memory.
    """
    p_train.train(
        parquet_file=str(parquet_file),
        output_dir=str(output_dir),
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        image_size=image_size,
        num_workers=num_workers,
        save_every=save_every,
        iteration_steps=iteration_steps,
        streaming=streaming,
    )


@app.command
def inference_dataset(
    checkpoint: str,
    parquet_file: str = "dataset.parquet",
    num_samples: int = 5,
    image_size: int = 512,
    output_dir: str = "./inference_outputs",
    iteration_steps: int = 10,
):
    """Inference on dataset samples."""
    inference.inference_from_dataset(
        checkpoint_path=checkpoint,
        parquet_file=parquet_file,
        num_samples=num_samples,
        image_size=image_size,
        output_dir=output_dir,
        iteration_steps=iteration_steps,
    )


@app.command
def inference_files(
    checkpoint: str,
    rgb_path: str,
    mask_path: str,
    timestep: float = 0.0,
    output_path: str = "./denoised_mask.png",
    image_size: int = 512,
    iteration_steps: int = 10,
):
    """Inference on RGB and mask image files."""
    inference.inference_from_files(
        checkpoint_path=checkpoint,
        rgb_path=rgb_path,
        mask_path=mask_path,
        timestep=timestep,
        output_path=output_path,
        image_size=image_size,
        iteration_steps=iteration_steps,
    )


if __name__ == "__main__":
    app.cli()
