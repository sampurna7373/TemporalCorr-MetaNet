"""Simple Decoder for TemporalCorr-MetaNet.

This module implements a simple upsampling decoder that produces
the final change detection mask from fused temporal features.

Architecture (from Section 5.7 of technical document):
- 3-stage upsampling: 64×64 → 128×128 → 256×256 → 512×512
- Bilinear interpolation + refinement convolutions
- Channel reduction: 256 → 128 → 64 → 32 → 1

Parameter count: ~80K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class UpsampleBlock(nn.Module):
    """Upsampling block with bilinear interpolation and refinement."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.refine(x)
        return x


class SimpleDecoder(nn.Module):
    """Simple upsampling decoder for change mask prediction.
    
    Takes fused temporal features and produces a binary change mask
    through progressive upsampling and channel reduction.
    
    Args:
        in_channels: Number of input channels (default: 256)
        hidden_channels: List of hidden channel sizes (default: [128, 64, 32])
        num_classes: Number of output classes (default: 1 for binary)
    
    Input: (B, 256, 64, 64) - fused temporal features
    Output: (B, 1, 512, 512) - change probability map
    
    Parameter count: ~80K
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: Optional[List[int]] = None,
        num_classes: int = 1,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [128, 64, 32]
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Build decoder layers
        layers = []
        prev_channels = in_channels
        
        for out_ch in hidden_channels:
            layers.append(UpsampleBlock(prev_channels, out_ch))
            prev_channels = out_ch
        
        self.upsample_layers = nn.ModuleList(layers)
        
        # Final prediction head
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[-1], num_classes, kernel_size=1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of decoder.
        
        Args:
            x: Input features of shape (B, C, H, W)
        
        Returns:
            Change probability map of shape (B, 1, H*8, W*8)
        """
        # Progressive upsampling
        for layer in self.upsample_layers:
            x = layer(x)
        
        # Final prediction
        out = self.final(x)
        
        # Sigmoid for probability
        out = torch.sigmoid(out)
        
        return out
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class MultiScaleDecoder(nn.Module):
    """Multi-scale decoder with skip connections (optional advanced version).
    
    This decoder can optionally use skip connections from the backbone
    for better spatial detail preservation.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        skip_channels: Optional[List[int]] = None,
        hidden_channels: Optional[List[int]] = None,
        num_classes: int = 1,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [128, 64, 32]
        if skip_channels is None:
            skip_channels = [0, 0, 0]  # No skip connections by default
        
        self.use_skips = any(c > 0 for c in skip_channels)
        
        layers = []
        prev_channels = in_channels
        
        for i, out_ch in enumerate(hidden_channels):
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            layers.append(
                self._make_decode_block(prev_channels + skip_ch, out_ch)
            )
            prev_channels = out_ch
        
        self.decode_layers = nn.ModuleList(layers)
        
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], num_classes, kernel_size=1),
        )
    
    def _make_decode_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with optional skip connections.
        
        Args:
            x: Input features (B, C, H, W)
            skips: Optional list of skip features from encoder
        
        Returns:
            Output predictions (B, 1, H*8, W*8)
        """
        for i, layer in enumerate(self.decode_layers):
            if skips is not None and i < len(skips) and skips[i] is not None:
                # Resize skip to match x if needed
                skip = skips[i]
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(
                        skip, size=x.shape[2:], mode="bilinear", align_corners=False
                    )
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
        
        out = self.final(x)
        out = torch.sigmoid(out)
        
        return out
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test SimpleDecoder
    model = SimpleDecoder(in_channels=256)
    print(f"SimpleDecoder parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 256, 64, 64)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
    # Expected output:
    # SimpleDecoder parameters: ~80,000
    # Input shape: torch.Size([2, 256, 64, 64])
    # Output shape: torch.Size([2, 1, 512, 512])
    # Output range: [0.xxx, 0.xxx]
