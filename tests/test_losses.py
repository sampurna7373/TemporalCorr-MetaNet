"""Unit tests for loss functions."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.losses import (
    FocalLoss,
    DiceLoss,
    CombinedLoss,
    CorrelationSparsityLoss,
    BoundarySmoothnessLoss,
    UncertaintyKLLoss,
    MetaLearningLoss,
)


class TestFocalLoss:
    """Test Focal Loss."""
    
    def test_output_shape(self):
        """Test output is scalar."""
        loss_fn = FocalLoss()
        pred = torch.sigmoid(torch.randn(4, 1, 64, 64))
        target = torch.randint(0, 2, (4, 1, 64, 64)).float()
        
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # Scalar
    
    def test_perfect_prediction(self):
        """Test loss is low for perfect predictions."""
        loss_fn = FocalLoss()
        target = torch.randint(0, 2, (2, 64, 64)).float()
        pred = target.clone()
        
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1
    
    def test_worst_prediction(self):
        """Test loss is high for inverted predictions."""
        loss_fn = FocalLoss()
        target = torch.ones(2, 64, 64)
        pred = torch.zeros(2, 64, 64)
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0.5
    
    def test_gradient(self):
        """Test gradients are computed."""
        loss_fn = FocalLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32, requires_grad=True))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None


class TestDiceLoss:
    """Test Dice Loss."""
    
    def test_output_range(self):
        """Test output is in [0, 1]."""
        loss_fn = DiceLoss()
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(pred, target)
        assert 0 <= loss.item() <= 1
    
    def test_perfect_overlap(self):
        """Test loss is 0 for perfect overlap."""
        loss_fn = DiceLoss()
        target = torch.ones(2, 64, 64)
        pred = target.clone()
        
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01


class TestCombinedLoss:
    """Test Combined Loss."""
    
    def test_combines_losses(self):
        """Test it combines focal and dice."""
        combined = CombinedLoss(focal_weight=1.0, dice_weight=1.0)
        focal = FocalLoss()
        dice = DiceLoss()
        
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        combined_loss = combined(pred, target)
        focal_loss = focal(pred, target)
        dice_loss = dice(pred, target)
        
        expected = focal_loss + dice_loss
        assert torch.isclose(combined_loss, expected, rtol=0.01)


class TestAuxiliaryLosses:
    """Test auxiliary loss functions."""
    
    def test_sparsity_loss(self):
        """Test correlation sparsity loss."""
        loss_fn = CorrelationSparsityLoss(weight=0.001)
        W = torch.randn(64, 64)
        
        loss = loss_fn(W)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_smoothness_loss(self):
        """Test boundary smoothness loss."""
        loss_fn = BoundarySmoothnessLoss(weight=0.0005)
        pred = torch.rand(2, 1, 64, 64)
        
        loss = loss_fn(pred)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_uncertainty_loss(self):
        """Test uncertainty KL loss."""
        loss_fn = UncertaintyKLLoss(weight=0.0002)
        uncertainty = torch.rand(4, 1)
        
        loss = loss_fn(uncertainty)
        assert loss.dim() == 0


class TestMetaLearningLoss:
    """Test complete meta-learning loss."""
    
    def test_loss_breakdown(self):
        """Test loss returns breakdown."""
        loss_fn = MetaLearningLoss()
        
        outputs = {
            "pred": torch.sigmoid(torch.randn(2, 1, 64, 64)),
            "uncertainty": torch.rand(2, 1),
        }
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        W_temporal = torch.randn(64, 64)
        
        losses = loss_fn(outputs, target, W_temporal)
        
        assert "seg" in losses
        assert "smooth" in losses
        assert "uncertain" in losses
        assert "sparse" in losses
        assert "total" in losses
    
    def test_total_is_sum(self):
        """Test total equals sum of components."""
        loss_fn = MetaLearningLoss()
        
        outputs = {
            "pred": torch.sigmoid(torch.randn(2, 1, 32, 32)),
            "uncertainty": torch.rand(2, 1),
        }
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        W_temporal = torch.randn(64, 64)
        
        losses = loss_fn(outputs, target, W_temporal)
        
        expected = losses["seg"] + losses["smooth"] + losses["uncertain"] + losses["sparse"]
        assert torch.isclose(losses["total"], expected, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
