"""Unit tests for model architecture.

Validates:
- Parameter counts match technical document specs
- Forward pass shapes are correct
- Gradient flow works properly
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    LightweightBackbone,
    PTCF,
    ATSM,
    MCB,
    SimpleDecoder,
    TemporalCorrMetaNet,
)


class TestLightweightBackbone:
    """Test LightweightBackbone module."""
    
    def test_output_shape(self):
        """Test output shape is correct."""
        model = LightweightBackbone()
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        
        assert out.shape == (2, 64, 64, 64), f"Expected (2, 64, 64, 64), got {out.shape}"
    
    def test_param_count(self):
        """Test parameter count is approximately 160K."""
        model = LightweightBackbone()
        params = model.get_num_params()
        
        # Should be around 160K (allow 20% margin)
        assert 120_000 < params < 200_000, f"Expected ~160K params, got {params:,}"
    
    def test_different_input_sizes(self):
        """Test with different input sizes."""
        model = LightweightBackbone()
        
        for size in [256, 512, 1024]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            expected = size // 8
            assert out.shape == (1, 64, expected, expected)
    
    def test_gradient_flow(self):
        """Test gradients flow properly."""
        model = LightweightBackbone()
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestPTCF:
    """Test PTCF module."""
    
    def test_output_shape(self):
        """Test output shape is correct."""
        model = PTCF(in_channels=64, out_channels=256)
        f_t1 = torch.randn(2, 64, 64, 64)
        f_t2 = torch.randn(2, 64, 64, 64)
        out = model(f_t1, f_t2)
        
        assert out.shape == (2, 256, 64, 64), f"Expected (2, 256, 64, 64), got {out.shape}"
    
    def test_param_count(self):
        """Test parameter count is approximately 280K."""
        model = PTCF()
        params = model.get_num_params()
        
        # Should be around 280K (allow margin)
        assert 200_000 < params < 400_000, f"Expected ~280K params, got {params:,}"
    
    def test_temporal_weights_exist(self):
        """Test learnable temporal weights exist."""
        model = PTCF(in_channels=64)
        
        assert hasattr(model, "W_temporal")
        assert model.W_temporal.shape == (64, 64)
    
    def test_gradient_to_temporal_weights(self):
        """Test gradients flow to temporal weights."""
        model = PTCF()
        f_t1 = torch.randn(2, 64, 32, 32)
        f_t2 = torch.randn(2, 64, 32, 32)
        
        out = model(f_t1, f_t2)
        loss = out.sum()
        loss.backward()
        
        assert model.W_temporal.grad is not None


class TestATSM:
    """Test ATSM module."""
    
    def test_output_shape(self):
        """Test output shape matches input."""
        model = ATSM(in_channels=256, num_classes=2)
        x = torch.randn(2, 256, 64, 64)
        out = model(x)
        
        assert out.shape == x.shape
    
    def test_param_count(self):
        """Test parameter count is approximately 150K."""
        model = ATSM()
        params = model.get_num_params()
        
        assert 100_000 < params < 200_000, f"Expected ~150K params, got {params:,}"
    
    def test_temperature_params(self):
        """Test temperature parameters exist."""
        model = ATSM(num_classes=2)
        
        assert hasattr(model, "temperature")
        assert model.temperature.shape == (2,)


class TestMCB:
    """Test MCB module."""
    
    def test_output_shapes(self):
        """Test all outputs have correct shapes."""
        model = MCB(in_channels=256, num_classes=2)
        x = torch.randn(4, 256, 64, 64)
        refined, thresholds, uncertainty = model(x)
        
        assert refined.shape == (4, 64, 64, 64)  # C//4
        assert thresholds.shape == (4, 2)
        assert uncertainty.shape == (4, 1)
    
    def test_thresholds_in_range(self):
        """Test thresholds are in valid range."""
        model = MCB()
        x = torch.randn(2, 256, 64, 64)
        _, thresholds, _ = model(x)
        
        assert (thresholds >= 0).all() and (thresholds <= 1).all()
    
    def test_param_count(self):
        """Test parameter count is approximately 80K."""
        model = MCB()
        params = model.get_num_params()
        
        assert 50_000 < params < 120_000, f"Expected ~80K params, got {params:,}"


class TestSimpleDecoder:
    """Test SimpleDecoder module."""
    
    def test_output_shape(self):
        """Test output shape is 8x upsampled."""
        model = SimpleDecoder(in_channels=256)
        x = torch.randn(2, 256, 64, 64)
        out = model(x)
        
        assert out.shape == (2, 1, 512, 512)
    
    def test_output_range(self):
        """Test output is in [0, 1] after sigmoid."""
        model = SimpleDecoder()
        x = torch.randn(2, 256, 64, 64)
        out = model(x)
        
        assert (out >= 0).all() and (out <= 1).all()
    
    def test_param_count(self):
        """Test parameter count is approximately 80K."""
        model = SimpleDecoder()
        params = model.get_num_params()
        
        assert 50_000 < params < 150_000, f"Expected ~80K params, got {params:,}"


class TestTemporalCorrMetaNet:
    """Test complete model."""
    
    def test_full_forward(self):
        """Test complete forward pass."""
        model = TemporalCorrMetaNet()
        img_t1 = torch.randn(2, 3, 512, 512)
        img_t2 = torch.randn(2, 3, 512, 512)
        
        outputs = model(img_t1, img_t2)
        
        assert "pred" in outputs
        assert "thresholds" in outputs
        assert "uncertainty" in outputs
        assert outputs["pred"].shape == (2, 1, 512, 512)
    
    def test_total_param_count(self):
        """Test total parameter count is approximately 750K."""
        model = TemporalCorrMetaNet()
        params = model.get_num_params()
        
        # Should be around 750K (allow 20% margin)
        assert 600_000 < params < 900_000, f"Expected ~750K params, got {params:,}"
        
        # Print breakdown
        by_module = model.get_num_params(by_module=True)
        print("\nParameter breakdown:")
        for name, count in by_module.items():
            print(f"  {name}: {count:,}")
    
    def test_predict_method(self):
        """Test prediction method."""
        model = TemporalCorrMetaNet()
        img_t1 = torch.randn(1, 3, 256, 256)
        img_t2 = torch.randn(1, 3, 256, 256)
        
        pred = model.predict(img_t1, img_t2 threshold=0.5)
        
        assert pred.shape == (1, 1, 256, 256)
        assert set(pred.unique().tolist()).issubset({0.0, 1.0})
    
    def test_return_features(self):
        """Test returning intermediate features."""
        model = TemporalCorrMetaNet()
        img_t1 = torch.randn(1, 3, 256, 256)
        img_t2 = torch.randn(1, 3, 256, 256)
        
        outputs = model(img_t1, img_t2, return_features=True)
        
        assert "features" in outputs
        assert "f_t1" in outputs["features"]
        assert "z" in outputs["features"]
    
    def test_gradient_flow(self):
        """Test gradients flow through complete model."""
        model = TemporalCorrMetaNet()
        img_t1 = torch.randn(1, 3, 128, 128, requires_grad=True)
        img_t2 = torch.randn(1, 3, 128, 128, requires_grad=True)
        
        outputs = model(img_t1, img_t2)
        loss = outputs["pred"].sum()
        loss.backward()
        
        assert img_t1.grad is not None
        assert img_t2.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
