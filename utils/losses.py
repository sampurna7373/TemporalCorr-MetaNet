"""Loss functions for change detection.

Implements loss functions from Section 6.2 of technical document:
- Focal Loss for class imbalance (α=0.25, γ=2.0)
- Dice Loss for segmentation
- Auxiliary losses: correlation sparsity, boundary smoothness, uncertainty

Total loss: L = L_focal + λ₁L_sparse + λ₂L_smooth + λ₃L_uncertain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t (1-p_t)^γ log(p_t)
    
    Where:
        p_t = p if y=1, else 1-p
        α_t = α if y=1, else 1-α
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predictions (B, 1, H, W) or (B, H, W), values in [0, 1]
            targets: Ground truth (B, 1, H, W) or (B, H, W), values in {0, 1}
        
        Returns:
            Loss value
        """
        # Ensure same shape
        if inputs.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        elif inputs.dim() == 3 and targets.dim() == 4:
            inputs = inputs.unsqueeze(1)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Clip to prevent log(0)
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        
        # Compute focal loss
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    Args:
        smooth: Smoothing factor to prevent division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dice loss.
        
        Args:
            inputs: Predictions (B, 1, H, W) or (B, H, W)
            targets: Ground truth (B, 1, H, W) or (B, H, W)
        
        Returns:
            Loss value
        """
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss with focal and dice components.
    
    L = w_focal * L_focal + w_dice * L_dice
    
    Args:
        focal_weight: Weight for focal loss
        dice_weight: Weight for dice loss
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
    """
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss.
    
    Applies higher weight to positive (change) class to handle imbalance.
    
    Args:
        pos_weight: Weight for positive class
    """
    
    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Clip
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        
        # Weighted BCE
        loss = -(
            self.pos_weight * targets * torch.log(inputs)
            + (1 - targets) * torch.log(1 - inputs)
        )
        
        return loss.mean()


class CorrelationSparsityLoss(nn.Module):
    """Sparsity regularization for correlation weights.
    
    Encourages sparse temporal correlation weights in PTCF module.
    
    L_sparse = λ * ||W_temporal||_1
    
    Args:
        weight: Regularization weight (λ₁ = 0.001)
    """
    
    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight
    
    def forward(self, W_temporal: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss.
        
        Args:
            W_temporal: Correlation weight matrix (C, C)
        
        Returns:
            Sparsity loss
        """
        return self.weight * torch.norm(W_temporal, p=1)


class BoundarySmoothnessLoss(nn.Module):
    """Smoothness regularization for change boundaries.
    
    Total Variation loss to encourage smooth predictions.
    
    L_smooth = λ * (||∇_x pred||_1 + ||∇_y pred||_1)
    
    Args:
        weight: Regularization weight (λ₂ = 0.0005)
    """
    
    def __init__(self, weight: float = 0.0005):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss.
        
        Args:
            predictions: Prediction map (B, 1, H, W)
        
        Returns:
            Smoothness loss
        """
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)
        
        # Compute gradients
        diff_x = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]
        diff_y = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        
        # L1 norm
        tv_loss = torch.abs(diff_x).mean() + torch.abs(diff_y).mean()
        
        return self.weight * tv_loss


class UncertaintyKLLoss(nn.Module):
    """KL divergence regularization for uncertainty.
    
    Penalizes deviation from uniform distribution to prevent
    overconfident predictions.
    
    Args:
        weight: Regularization weight (λ₃ = 0.0002)
    """
    
    def __init__(self, weight: float = 0.0002):
        super().__init__()
        self.weight = weight
    
    def forward(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from uniform.
        
        Args:
            uncertainty: Uncertainty scores (B, 1)
        
        Returns:
            KL loss
        """
        # Target: uniform distribution (0.5)
        uniform = torch.full_like(uncertainty, 0.5)
        
        # Clip to prevent log(0)
        uncertainty = torch.clamp(uncertainty, 1e-7, 1 - 1e-7)
        
        # KL divergence
        kl = uncertainty * torch.log(uncertainty / uniform) + \
             (1 - uncertainty) * torch.log((1 - uncertainty) / (1 - uniform))
        
        return self.weight * kl.mean()


class MetaLearningLoss(nn.Module):
    """Complete loss function for meta-learning.
    
    Combines:
    - Main: Focal + Dice loss
    - Aux: Sparsity + Smoothness + Uncertainty
    
    Args:
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        dice_weight: Weight for dice loss
        sparsity_weight: Weight for correlation sparsity
        smoothness_weight: Weight for boundary smoothness
        uncertainty_weight: Weight for uncertainty regularization
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.5,
        sparsity_weight: float = 0.001,
        smoothness_weight: float = 0.0005,
        uncertainty_weight: float = 0.0002,
    ):
        super().__init__()
        self.combined = CombinedLoss(
            focal_weight=1.0,
            dice_weight=dice_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        self.sparsity = CorrelationSparsityLoss(sparsity_weight)
        self.smoothness = BoundarySmoothnessLoss(smoothness_weight)
        self.uncertainty = UncertaintyKLLoss(uncertainty_weight)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        W_temporal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with breakdown.
        
        Args:
            outputs: Model outputs with 'pred', 'uncertainty'
            targets: Ground truth masks
            W_temporal: Optional correlation weights for sparsity loss
        
        Returns:
            Dictionary with loss breakdown
        """
        losses = {}
        
        # Main segmentation loss
        losses["seg"] = self.combined(outputs["pred"], targets)
        
        # Auxiliary losses
        losses["smooth"] = self.smoothness(outputs["pred"])
        
        if "uncertainty" in outputs:
            losses["uncertain"] = self.uncertainty(outputs["uncertainty"])
        else:
            losses["uncertain"] = torch.tensor(0.0, device=outputs["pred"].device)
        
        if W_temporal is not None:
            losses["sparse"] = self.sparsity(W_temporal)
        else:
            losses["sparse"] = torch.tensor(0.0, device=outputs["pred"].device)
        
        # Total
        losses["total"] = (
            losses["seg"] + losses["smooth"] + 
            losses["uncertain"] + losses["sparse"]
        )
        
        return losses


def create_loss_function(config=None) -> MetaLearningLoss:
    """Factory function to create loss from config."""
    if config is None:
        return MetaLearningLoss()
    
    return MetaLearningLoss(
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        dice_weight=0.5,
        sparsity_weight=config.correlation_sparsity_weight,
        smoothness_weight=config.boundary_smoothness_weight,
        uncertainty_weight=config.uncertainty_reg_weight,
    )


if __name__ == "__main__":
    # Test loss functions
    pred = torch.sigmoid(torch.randn(4, 1, 64, 64))
    target = torch.randint(0, 2, (4, 1, 64, 64)).float()
    
    # Test Focal Loss
    focal = FocalLoss()
    loss = focal(pred, target)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test Dice Loss
    dice = DiceLoss()
    loss = dice(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    combined = CombinedLoss()
    loss = combined(pred, target)
    print(f"Combined Loss: {loss.item():.4f}")
    
    # Test Meta Learning Loss
    outputs = {"pred": pred, "uncertainty": torch.rand(4, 1)}
    W_temporal = torch.randn(64, 64)
    
    meta_loss = MetaLearningLoss()
    losses = meta_loss(outputs, target, W_temporal)
    print(f"\nMeta Learning Loss breakdown:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
