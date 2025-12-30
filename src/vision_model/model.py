"""U-Net model architecture for diffusion model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) tensor of timesteps
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # GroupNorm requires num_groups to divide channels evenly
        # Use min(8, channels) groups, but ensure it divides evenly
        def get_num_groups(channels: int) -> int:
            if channels <= 0:
                return 1
            # Always use 1 group as minimum
            if channels == 1:
                return 1
            # Try to use up to 8 groups, but must divide evenly
            max_groups = min(8, channels)
            # Find the largest divisor of channels that's <= max_groups
            for groups in range(max_groups, 0, -1):
                if channels % groups == 0:
                    return groups
            # Fallback: always return 1 if no divisor found (shouldn't happen)
            return 1

        in_groups = get_num_groups(in_channels)
        out_groups = get_num_groups(out_channels)

        # Double-check that groups divide channels evenly (safety check)
        if in_channels % in_groups != 0:
            raise ValueError(
                f"GroupNorm error: in_channels={in_channels} not divisible by "
                f"in_groups={in_groups}. This should not happen!"
            )
        if out_channels % out_groups != 0:
            raise ValueError(
                f"GroupNorm error: out_channels={out_channels} not divisible by "
                f"out_groups={out_groups}. This should not happen!"
            )

        self.block1 = nn.Sequential(
            nn.GroupNorm(in_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(out_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    """U-Net architecture for diffusion model with RGB conditioning.

    Takes RGB image as conditioning and denoises a mask.
    Input: RGB image (3 channels) + noisy mask (1 channel) = 4 channels total
    Output: Denoised mask (1 channel)
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        mask_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        channel_multipliers: tuple[int, int, int, int] = (8, 8, 12, 16),
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.rgb_channels = rgb_channels
        self.mask_channels = mask_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Encoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Input channels: RGB (3) + mask (1) = 4
        in_channels = rgb_channels + mask_channels
        channels = [in_channels] + [base_channels * m for m in channel_multipliers]

        # Encoder blocks
        for i in range(len(channels) - 1):
            self.encoder.append(
                ResidualBlock(channels[i], channels[i + 1], time_emb_dim)
            )

        # Bottleneck
        self.bottleneck = ResidualBlock(channels[-1], channels[-1], time_emb_dim)

        # Decoder blocks
        # After encoder: skip_connections stored in forward order
        # After bottleneck: x has channels[-1] channels
        # Decoder processes in reverse: pops skip connections from end
        # Each decoder block: upsampled features + skip connection -> output
        for j in range(len(channels) - 1):
            # j goes 0 to len-2 (forward order for decoder list)
            # skip_idx goes len-2 to 0 (reverse order, matching pop order)
            skip_idx = len(channels) - 2 - j
            # After upsampling: channels[skip_idx + 1] channels
            # Skip connection: channels[skip_idx + 1] channels (from encoder output at that level)
            # Combined input: channels[skip_idx + 1] * 2
            # Output: channels[skip_idx]
            input_channels = channels[skip_idx + 1] * 2
            output_channels = channels[skip_idx]
            self.decoder.append(
                ResidualBlock(input_channels, output_channels, time_emb_dim)
            )

        # Output layer - outputs single channel mask
        # The decoder outputs channels[0] channels (same as input channels)
        output_channels = channels[0]  # This is the actual output from decoder

        # GroupNorm requires num_groups to divide channels evenly
        def get_num_groups(channels: int) -> int:
            if channels <= 0:
                return 1
            if channels == 1:
                return 1
            max_groups = min(8, channels)
            for groups in range(max_groups, 0, -1):
                if channels % groups == 0:
                    return groups
            return 1

        out_norm_groups = get_num_groups(output_channels)

        # Validate
        if output_channels % out_norm_groups != 0:
            raise ValueError(
                f"Output normalization error: output_channels={output_channels} not divisible by "
                f"out_norm_groups={out_norm_groups}. This should not happen!"
            )

        self.out_norm = nn.GroupNorm(out_norm_groups, output_channels)
        self.out_conv = nn.Conv2d(output_channels, out_channels, 3, padding=1)

    def forward(
        self, rgb: torch.Tensor, noisy_mask: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) RGB conditioning image
            noisy_mask: (B, 1, H, W) noisy mask
            timestep: (B,) timestep tensor
        Returns:
            (B, 1, H, W) predicted clean mask
        """
        # Concatenate RGB and mask along channel dimension
        x = torch.cat([rgb, noisy_mask], dim=1)  # (B, 4, H, W)

        # Time embedding
        t = self.time_mlp(timestep)

        # Encoder
        skip_connections = []
        for block in self.encoder:
            x = block(x, t)
            skip_connections.append(x)
            x = F.avg_pool2d(x, 2)

        # Bottleneck
        x = self.bottleneck(x, t)

        # Decoder
        for block in self.decoder:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, t)

        # Output
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)  # (B, 1, H, W)

        return x
