"""
Modality-aware fusion module for RGB and IR features.

Supports RGB+IR, RGB-only, and IR-only modes with learnable fusion.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModalityAwareFusion(nn.Module):
    """
    Modality-aware fusion for RGB and IR features.

    Supports:
    - RGB + IR: Both modalities contribute
    - RGB only: RGB embedding dominates
    - IR only: IR embedding dominates

    Uses learnable projections and gated fusion.
    Output dimension is fixed (512-D) regardless of missing modalities.
    """

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 256,
                 output_dim: int = 512):
        """
        Initialize modality-aware fusion.

        Args:
            input_dim: Input feature dimension (2048 for ResNet50)
            hidden_dim: Hidden dimension for projections
            output_dim: Output fusion dimension (512)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Projections for each modality
        self.rgb_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.ir_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Fusion to output dimension
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

        # Single-modality adaptation layers
        self.rgb_adapt = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

        self.ir_adapt = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

        logger.info(f"Modality-aware fusion: input={input_dim}, hidden={hidden_dim}, output={output_dim}")

    def forward(self,
                rgb_features: torch.Tensor,
                ir_features: Optional[torch.Tensor] = None,
                rgb_available: bool = True,
                ir_available: bool = True) -> torch.Tensor:
        """
        Fuse RGB and IR features with modality awareness.

        Args:
            rgb_features: RGB features (B, 2048)
            ir_features: IR features (B, 2048) or None
            rgb_available: Whether RGB modality is available
            ir_available: Whether IR modality is available

        Returns:
            Fused features (B, 512)
        """
        # Project RGB features
        if rgb_available:
            rgb_embed = self.rgb_projection(rgb_features)
        else:
            # Zero embedding if unavailable
            rgb_embed = torch.zeros(rgb_features.shape[0], self.hidden_dim, device=rgb_features.device)

        # Project IR features
        if ir_available and ir_features is not None:
            ir_embed = self.ir_projection(ir_features)
        else:
            # Zero embedding if unavailable
            ir_embed = torch.zeros(rgb_features.shape[0], self.hidden_dim, device=rgb_features.device)

        # Fusion based on availability
        if rgb_available and ir_available:
            # Both modalities: concatenate and fuse
            combined = torch.cat([rgb_embed, ir_embed], dim=1)
            fused = self.fusion_layer(combined)
        elif rgb_available:
            # RGB only: adapt RGB embedding
            fused = self.rgb_adapt(rgb_embed)
        elif ir_available:
            # IR only: adapt IR embedding
            fused = self.ir_adapt(ir_embed)
        else:
            # Neither available: return zeros
            fused = torch.zeros(rgb_features.shape[0], self.output_dim, device=rgb_features.device)

        return fused

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim
