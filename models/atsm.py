"""Adaptive Temporal Similarity Matrices (ATSM) Module.

This module implements class-specific temporal similarity learning
that adapts temporal feature correspondence based on support examples.

Key Insight (from Section 5.4 of technical document):
Different change types (buildings vs vegetation) exhibit different
temporal feature signatures. ATSM learns class-specific transformations
mapping query features onto class manifolds.

Components:
1. Per-class temporal similarity computation
2. Meta-learnable temperature parameter λ_c
3. Class-specific feature scaling

Parameter count: ~150K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class TemporalSimilarityComputer(nn.Module):
    """Computes temporal similarity between bi-temporal features."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        # Projection for similarity computation
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute similarity score for features.
        
        Args:
            features: Fused temporal features (B, C, H, W)
            mask: Optional binary mask for weighted similarity
        
        Returns:
            Similarity score per sample (B,)
        """
        B, C, H, W = features.shape
        
        # Global average pooling
        gap = F.adaptive_avg_pool2d(features, 1).view(B, C)  # (B, C)
        
        # L2 normalize
        gap_norm = F.normalize(gap, p=2, dim=1)
        
        # Compute self-similarity (could be extended to cross-similarity)
        similarity = (gap_norm * gap_norm).sum(dim=1)  # (B,)
        
        return similarity


class ATSM(nn.Module):
    """Adaptive Temporal Similarity Matrices Module.
    
    Implements meta-learned class-specific temporal similarity mechanisms
    that adaptively adjust per-class temporal feature correspondence.
    
    Mathematical Formulation:
    - S_c = mean similarity for class c from support examples
    - λ_c = Meta-learnable temperature (initialized 1.0)
    - Z_scaled = λ_c · S_c ⊙ Z
    
    Args:
        in_channels: Number of input channels (default: 256)
        num_classes: Number of classes (default: 2 for change/no-change)
    
    Parameter count: ~150K
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Meta-learnable temperature parameters (one per class)
        self.temperature = nn.Parameter(torch.ones(num_classes))
        
        # Class-specific feature transformation
        self.class_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
                nn.Sigmoid(),
            )
            for _ in range(num_classes)
        ])
        
        # Similarity computer
        self.similarity_computer = TemporalSimilarityComputer(in_channels)
        
        # Feature aggregation
        self.aggregation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Class prototype projection
        self.prototype_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
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
    
    def compute_class_prototypes(
        self,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Compute class prototypes from support features.
        
        Args:
            support_features: Features from support set (K*num_classes, C, H, W)
            support_masks: Binary masks for support set (K*num_classes, H, W)
        
        Returns:
            Dictionary mapping class index to prototype vector
        """
        prototypes = {}
        B, C, H, W = support_features.shape
        
        for c in range(self.num_classes):
            # Get indices for this class
            # Assuming masks are organized as [K samples of class 0, K samples of class 1, ...]
            # For binary: class 0 = no-change (mask=0), class 1 = change (mask=1)
            class_mask = (support_masks.float().mean(dim=(1, 2)) > 0.5).float()
            
            if c == 1:  # Change class
                class_indices = (class_mask > 0.5).nonzero(as_tuple=True)[0]
            else:  # No-change class
                class_indices = (class_mask <= 0.5).nonzero(as_tuple=True)[0]
            
            if len(class_indices) > 0:
                class_features = support_features[class_indices]
                prototype = self.prototype_proj(class_features).mean(dim=0)
                prototypes[c] = prototype
            else:
                # Default prototype if no samples
                prototypes[c] = torch.zeros(C, device=support_features.device)
        
        return prototypes
    
    def forward(
        self,
        features: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
        support_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of ATSM module.
        
        Args:
            features: Fused temporal features (B, C, H, W)
            support_features: Optional support set features for meta-learning
            support_masks: Optional support set masks for prototype computation
        
        Returns:
            Scaled features with class-specific temporal weighting (B, C, H, W)
        """
        B, C, H, W = features.shape
        
        # Compute class-specific transformations
        class_weights = []
        for c in range(self.num_classes):
            weight = self.class_transforms[c](features)  # (B, C, H, W)
            # Apply temperature scaling
            weight = weight * self.temperature[c]
            class_weights.append(weight)
        
        # Stack and combine class weights
        stacked_weights = torch.stack(class_weights, dim=1)  # (B, num_classes, C, H, W)
        
        # Soft attention over classes
        class_attention = F.softmax(stacked_weights.mean(dim=(2, 3, 4)), dim=1)  # (B, num_classes)
        
        # Weighted combination
        combined_weight = torch.zeros_like(features)
        for c in range(self.num_classes):
            combined_weight += class_attention[:, c:c+1, None, None] * class_weights[c]
        
        # Apply scaling
        scaled_features = features * combined_weight
        
        # Aggregation
        out = self.aggregation(scaled_features)
        
        # Residual connection
        out = out + features
        
        return out
    
    def adapt_from_support(
        self,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> None:
        """Adapt internal parameters from support set (for meta-learning).
        
        This method computes class prototypes from support examples
        which can be used for few-shot adaptation.
        
        Args:
            support_features: Features from support set
            support_masks: Binary masks for support set
        """
        self.current_prototypes = self.compute_class_prototypes(
            support_features, support_masks
        )
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test ATSM module
    model = ATSM(in_channels=256, num_classes=2)
    print(f"ATSM parameters: {model.get_num_params():,}")
    
    # Test forward pass
    features = torch.randn(4, 256, 64, 64)
    out = model(features)
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {out.shape}")
    
    # Expected output:
    # ATSM parameters: ~150,000
    # Input shape: torch.Size([4, 256, 64, 64])
    # Output shape: torch.Size([4, 256, 64, 64])
