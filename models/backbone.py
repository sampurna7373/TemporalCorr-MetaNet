"""Lightweight CNN Backbone for TemporalCorr-MetaNet.

This module implements a simple 3-layer CNN backbone (~160K parameters)
that extracts features from bi-temporal satellite images.

Architecture (from Section 5.2 of technical document):
- Layer 1: 512×512 → 256×256, channels 3→32
- Layer 2: 256×256 → 128×128, channels 32→48
- Layer 3: 128×128 → 64×64, channels 48→64
- Output: F_T₁, F_T₂ ∈ ℝ^(64×64×64)

Design Rationale:
- Simple 3-layer CNN (no ResNet/DenseNet complexity)
- BatchNorm + ReLU for stable training
- Shared weights between T₁ and T₂ images
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d + BatchNorm + ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_bn: bool = True,
        use_relu: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightBackbone(nn.Module):
    """Lightweight CNN backbone for feature extraction.
    
    A simple 3-layer CNN that reduces spatial dimensions by 8x while
    extracting discriminative features for change detection.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        channels: List of output channels for each layer (default: [32, 48, 64])
    
    Input: (B, C, H, W) where H=W=512
    Output: (B, 64, H/8, W/8) where H/8=W/8=64
    
    Parameter count: ~160K
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = None,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 48, 64]
        
        assert len(channels) == 3, "Expected 3 channel values for 3-layer backbone"
        
        # Layer 1: 512×512 → 256×256, channels: 3 → 32
        self.conv1 = ConvBlock(
            in_channels, channels[0], kernel_size=3, stride=2, padding=1
        )
        
        # Layer 2: 256×256 → 128×128, channels: 32 → 48
        self.conv2 = ConvBlock(
            channels[0], channels[1], kernel_size=3, stride=2, padding=1
        )
        
        # Layer 3: 128×128 → 64×64, channels: 48 → 64
        self.conv3 = ConvBlock(
            channels[1], channels[2], kernel_size=3, stride=2, padding=1
        )
        
        self.out_channels = channels[2]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input image.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Feature tensor of shape (B, 64, H/8, W/8)
        """
        x = self.conv1(x)  # (B, 32, H/2, W/2)
        x = self.conv2(x)  # (B, 48, H/4, W/4)
        x = self.conv3(x)  # (B, 64, H/8, W/8)
        return x
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_backbone(config=None) -> LightweightBackbone:
    """Factory function to create backbone from config.
    
    Args:
        config: ModelConfig object or None for defaults
    
    Returns:
        LightweightBackbone instance
    """
    if config is None:
        return LightweightBackbone()
    
    return LightweightBackbone(
        in_channels=config.in_channels,
        channels=config.backbone_channels,
    )


if __name__ == "__main__":
    # Test backbone
    model = LightweightBackbone()
    print(f"Backbone parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Expected output:
    # Backbone parameters: ~160,000
    # Input shape: torch.Size([2, 3, 512, 512])
    # Output shape: torch.Size([2, 64, 64, 64])
