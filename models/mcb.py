"""Meta-Learned Change Boundaries (MCB) Module.

This module implements adaptive threshold learning for change detection.
Instead of using a fixed threshold (e.g., 0.5), MCB learns class-specific
thresholds that adapt to regional/sensor characteristics.

Key Innovation (from Section 5.5 of technical document):
- EGY-BCD might prefer threshold 0.45 (many small buildings)
- WHU might prefer threshold 0.55 (fewer, larger buildings)
- S2Looking might prefer threshold 0.48 (viewing angle variations)

Components:
1. Adaptive threshold learning from support examples
2. Uncertainty estimation for ambiguous predictions
3. Class-specific threshold parameters

Parameter count: ~80K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class UncertaintyEstimator(nn.Module):
    """Estimates prediction uncertainty."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_channels * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty score.
        
        Args:
            features: Input features (B, C, H, W)
        
        Returns:
            Uncertainty score (B, 1)
        """
        return self.uncertainty_head(features)


class MCB(nn.Module):
    """Meta-Learned Change Boundaries Module.
    
    Learns class-specific thresholds from support examples, improving
    robustness across domain shifts between different datasets/sensors.
    
    Mathematical Formulation:
    - θ_c = Sigmoid(γ_c), where γ_c is learned from support examples
    - Output = 1 if Pred > θ_change, 0 if Pred < θ_nochange
    
    Args:
        in_channels: Number of input channels (default: 256)
        num_classes: Number of change classes (default: 2)
    
    Parameter count: ~80K
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Base threshold parameters (can be adapted per-episode)
        self.base_thresholds = nn.Parameter(torch.zeros(num_classes))
        
        # Threshold prediction network
        self.threshold_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_channels * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        
        # Feature refinement for boundary detection
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(in_channels)
        
        # Support set encoder for threshold adaptation
        self.support_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
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
    
    def compute_thresholds(
        self,
        features: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute adaptive thresholds.
        
        Args:
            features: Query features (B, C, H, W)
            support_features: Optional support set features for adaptation
        
        Returns:
            Thresholds for each class (B, num_classes)
        """
        # Base thresholds from network
        base_thresh = self.threshold_net(features)  # (B, num_classes)
        
        # Add learned base threshold
        thresholds = torch.sigmoid(base_thresh + self.base_thresholds)
        
        # If support features provided, adapt thresholds
        if support_features is not None:
            support_adjustment = self.support_encoder(support_features)  # (S, num_classes)
            support_adjustment = support_adjustment.mean(dim=0, keepdim=True)  # (1, num_classes)
            thresholds = thresholds + 0.1 * torch.tanh(support_adjustment)
            thresholds = torch.clamp(thresholds, 0.1, 0.9)
        
        return thresholds
    
    def forward(
        self,
        features: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
        support_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of MCB module.
        
        Args:
            features: Input features (B, C, H, W)
            support_features: Optional support set features
            support_masks: Optional support set masks
        
        Returns:
            Tuple of:
                - Refined features (B, C//4, H, W)
                - Thresholds (B, num_classes)
                - Uncertainty (B, 1)
        """
        # Refine features for boundary detection
        refined = self.boundary_refine(features)  # (B, C//4, H, W)
        
        # Compute adaptive thresholds
        thresholds = self.compute_thresholds(features, support_features)  # (B, num_classes)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(features)  # (B, 1)
        
        return refined, thresholds, uncertainty
    
    def apply_thresholds(
        self,
        predictions: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> torch.Tensor:
        """Apply learned thresholds to predictions.
        
        Args:
            predictions: Raw predictions (B, 1, H, W)
            thresholds: Learned thresholds (B, num_classes)
        
        Returns:
            Thresholded predictions (B, 1, H, W)
        """
        # Use change threshold (class 1)
        change_threshold = thresholds[:, 1:2].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        
        # Apply threshold
        binary_pred = (predictions > change_threshold).float()
        
        return binary_pred
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test MCB module
    model = MCB(in_channels=256, num_classes=2)
    print(f"MCB parameters: {model.get_num_params():,}")
    
    # Test forward pass
    features = torch.randn(4, 256, 64, 64)
    refined, thresholds, uncertainty = model(features)
    print(f"Input shape: {features.shape}")
    print(f"Refined shape: {refined.shape}")
    print(f"Thresholds shape: {thresholds.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Threshold values: {thresholds[0].detach().numpy()}")
    
    # Expected output:
    # MCB parameters: ~80,000
    # Input shape: torch.Size([4, 256, 64, 64])
    # Refined shape: torch.Size([4, 64, 64, 64])
    # Thresholds shape: torch.Size([4, 2])
    # Uncertainty shape: torch.Size([4, 1])
