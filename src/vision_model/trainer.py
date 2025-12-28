"""Trainer class for Bernoulli diffusion model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BernoulliDiffusionModel:
    """Bernoulli diffusion model trainer."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        rgb = batch["rgb"].to(self.device)
        noisy_mask = batch["noisy_mask"].to(self.device)
        clean_mask = batch["clean_mask"].to(self.device)
        timestep = batch["timestep"].to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                pred_clean_mask = self.model(rgb, noisy_mask, timestep)
                loss = F.mse_loss(pred_clean_mask, clean_mask)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred_clean_mask = self.model(rgb, noisy_mask, timestep)
            loss = F.mse_loss(pred_clean_mask, clean_mask)
            loss.backward()
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "pred": pred_clean_mask.detach(),
            "target": clean_mask.detach(),
        }

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["loss"]
