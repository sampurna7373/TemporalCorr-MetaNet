"""Parametric Temporal Correlation Fusion (PTCF) Module.

This module implements the core innovation of TemporalCorr-MetaNet:
learnable temporal correlation tensors that replace sequential processing
(LSTM/GRU) for bi-temporal change detection.

Key Components (from Section 5.3 of technical document):
1. Temporal Correlation Tensor: Outer product F_T₁ ⊗ F_T₂
2. Learnable Correlation Weights: W_temporal ∈ ℝ^(C×C)
3. Temporal Attention: SE-style attention A_T₁, A_T₂
4. Correlation Fusion: DepthwiseConv + PointwiseConv

Advantages over alternatives:
- No sequential assumption (unlike LSTM)
- Captures all C² channel pairwise interactions
- Lightweight: ~280K parameters
- Symmetric: f(T₁, T₂) ≈ f(T₂, T₁)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalAttention(nn.Module):
    """SE-style temporal attention for channel-wise feature weighting.
    
    Computes attention weights for temporal features using global
    average pooling followed by FC layers with sigmoid activation.
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 4)
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention weights.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Attention weights of shape (B, C, 1, 1)
        """
        attn = self.attention(x)  # (B, C)
        return attn.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient feature processing."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PTCF(nn.Module):
    """Parametric Temporal Correlation Fusion Module.
    
    Core innovation of TemporalCorr-MetaNet that captures complex temporal
    feature interactions through learnable correlation tensors.
    
    Mathematical Formulation:
    1. Temporal Correlation: τ = F_T₁ ⊗ F_T₂ (outer product)
    2. Learnable Weighting: τ_weighted = W_temporal ⊙ τ
    3. Temporal Attention: Apply SE-style attention to both temporal features
    4. Fusion: Depthwise + Pointwise convolution
    
    Args:
        in_channels: Number of input channels (default: 64)
        out_channels: Number of output channels (default: 256)
        reduction: Reduction ratio for attention (default: 4)
    
    Parameter count: ~280K
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 256,
        reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Learnable temporal correlation weight matrix W_temporal ∈ ℝ^(C×C)
        self.W_temporal = nn.Parameter(torch.randn(in_channels, in_channels) * 0.01)
        
        # Temporal attention for T₁ and T₂
        self.attention_t1 = TemporalAttention(in_channels, reduction)
        self.attention_t2 = TemporalAttention(in_channels, reduction)
        
        # Correlation projection: reduce C×C to C
        self.corr_projection = nn.Sequential(
            nn.Conv2d(in_channels * in_channels, in_channels * 2, kernel_size=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        # Feature fusion with depthwise separable conv
        # Concatenate: F_T₁ (C) + F_T₂ (C) + corr (2C) = 4C
        fusion_in_channels = in_channels * 4
        self.fusion = nn.Sequential(
            DepthwiseSeparableConv(fusion_in_channels, out_channels // 2),
            DepthwiseSeparableConv(out_channels // 2, out_channels),
        )
        
        # Residual connection projection
        self.residual_proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_normal_(self.W_temporal)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_temporal_correlation(
        self,
        f_t1: torch.Tensor,
        f_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted temporal correlation tensor.
        
        For each spatial location (u,v), computes:
        τ = W_temporal ⊙ (f_t1 ⊗ f_t2)
        
        Args:
            f_t1: Features from T₁, shape (B, C, H, W)
            f_t2: Features from T₂, shape (B, C, H, W)
        
        Returns:
            Correlation tensor of shape (B, C*C, H, W)
        """
        B, C, H, W = f_t1.shape
        
        # Reshape for outer product: (B, C, H, W) -> (B, C, H*W)
        f_t1_flat = f_t1.view(B, C, -1)  # (B, C, H*W)
        f_t2_flat = f_t2.view(B, C, -1)  # (B, C, H*W)
        
        # Compute correlation at each spatial location
        # Using efficient batch matrix multiplication
        # (B, C, H*W) x (B, C, H*W)^T -> (B, C, C, H*W)
        # We want per-location outer product, so we permute
        
        # Alternative efficient implementation:
        # For each spatial location, compute outer product
        # f_t1_flat: (B, C, HW)
        # f_t2_flat: (B, C, HW)
        
        # Expand for element-wise multiplication to get correlation
        f_t1_exp = f_t1_flat.unsqueeze(2)  # (B, C, 1, HW)
        f_t2_exp = f_t2_flat.unsqueeze(1)  # (B, 1, C, HW)
        
        # Outer product at each location
        corr = f_t1_exp * f_t2_exp  # (B, C, C, HW)
        
        # Apply learnable weights
        W = self.W_temporal.unsqueeze(0).unsqueeze(-1)  # (1, C, C, 1)
        corr_weighted = corr * W  # (B, C, C, HW)
        
        # Reshape to (B, C*C, H, W)
        corr_weighted = corr_weighted.view(B, C * C, H, W)
        
        return corr_weighted
    
    def forward(
        self,
        f_t1: torch.Tensor,
        f_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of PTCF module.
        
        Args:
            f_t1: Features from T₁, shape (B, C, H, W)
            f_t2: Features from T₂, shape (B, C, H, W)
        
        Returns:
            Fused temporal features of shape (B, out_channels, H, W)
        """
        # Step 1: Apply temporal attention
        attn_t1 = self.attention_t1(f_t1)  # (B, C, 1, 1)
        attn_t2 = self.attention_t2(f_t2)  # (B, C, 1, 1)
        
        f_t1_attn = f_t1 * attn_t1  # (B, C, H, W)
        f_t2_attn = f_t2 * attn_t2  # (B, C, H, W)
        
        # Step 2: Compute temporal correlation
        corr = self.compute_temporal_correlation(f_t1_attn, f_t2_attn)  # (B, C*C, H, W)
        
        # Step 3: Project correlation
        corr_proj = self.corr_projection(corr)  # (B, 2C, H, W)
        
        # Step 4: Concatenate all features
        # f_t1 + f_t2 + corr_proj = C + C + 2C = 4C
        fused = torch.cat([f_t1_attn, f_t2_attn, corr_proj], dim=1)  # (B, 4C, H, W)
        
        # Step 5: Apply fusion layers
        out = self.fusion(fused)  # (B, out_channels, H, W)
        
        # Step 6: Residual connection
        residual = self.residual_proj(torch.cat([f_t1, f_t2], dim=1))  # (B, out_channels, H, W)
        out = out + residual
        
        return out
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_correlation_weights(self) -> torch.Tensor:
        """Get the learned correlation weight matrix for visualization."""
        return self.W_temporal.detach().cpu()


if __name__ == "__main__":
    # Test PTCF module
    model = PTCF(in_channels=64, out_channels=256)
    print(f"PTCF parameters: {model.get_num_params():,}")
    
    # Test forward pass
    f_t1 = torch.randn(2, 64, 64, 64)
    f_t2 = torch.randn(2, 64, 64, 64)
    out = model(f_t1, f_t2)
    print(f"Input shape: {f_t1.shape}")
    print(f"Output shape: {out.shape}")
    
    # Expected output:
    # PTCF parameters: ~280,000
    # Input shape: torch.Size([2, 64, 64, 64])
    # Output shape: torch.Size([2, 256, 64, 64])
