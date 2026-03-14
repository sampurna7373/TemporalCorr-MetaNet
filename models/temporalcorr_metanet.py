"""TemporalCorr-MetaNet: Complete Model Assembly.

This module assembles all components into the complete TemporalCorr-MetaNet
architecture for few-shot remote sensing change detection.

Architecture Overview (from Section 5.1):
┌─────────────────────────────────────────────────────────────┐
│ Input: Bi-temporal image pair (I_T₁, I_T₂) ∈ ℝ^(3×512×512) │
│         │                                                    │
│         ├─ Shared Lightweight CNN Backbone (160K params)    │
│         │  ├─ I_T₁ → Conv blocks → F_T₁ ∈ ℝ^(64×64×64)     │
│         │  └─ I_T₂ → Conv blocks → F_T₂ ∈ ℝ^(64×64×64)     │
│         │                                                    │
│         ├─ PTCF Module (280K params)                        │
│         │  └─ Temporal correlation → Z ∈ ℝ^(256×64×64)      │
│         │                                                    │
│         ├─ ATSM Module (150K params)                        │
│         │  └─ Class-specific scaling                        │
│         │                                                    │
│         ├─ MCB Module (80K params)                          │
│         │  └─ Adaptive thresholds + uncertainty             │
│         │                                                    │
│         ├─ Decoder (80K params)                             │
│         │  └─ Z → upsampling → Ŷ ∈ {0,1}^(512×512)         │
│         │                                                    │
│         Output: Change detection mask Ŷ                      │
└─────────────────────────────────────────────────────────────┘

Total Parameters: ~750K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .backbone import LightweightBackbone
from .ptcf import PTCF
from .atsm import ATSM
from .mcb import MCB
from .decoder import SimpleDecoder


class TemporalCorrMetaNet(nn.Module):
    """Complete TemporalCorr-MetaNet for few-shot change detection.
    
    Assembles all components into a unified model:
    - Shared backbone for bi-temporal feature extraction
    - PTCF for temporal correlation fusion
    - ATSM for adaptive temporal similarity
    - MCB for meta-learned change boundaries
    - Decoder for mask prediction
    
    Args:
        in_channels: Input image channels (default: 3)
        backbone_channels: Backbone channel progression (default: [32, 48, 64])
        ptcf_out_channels: PTCF output channels (default: 256)
        num_classes: Number of classes for ATSM/MCB (default: 2)
        decoder_channels: Decoder channel progression (default: [128, 64, 32])
    
    Total parameters: ~750K
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: list = None,
        ptcf_out_channels: int = 256,
        num_classes: int = 2,
        decoder_channels: list = None,
    ):
        super().__init__()
        
        if backbone_channels is None:
            backbone_channels = [32, 48, 64]
        if decoder_channels is None:
            decoder_channels = [128, 64, 32]
        
        backbone_out = backbone_channels[-1]
        
        # Shared backbone (~160K params)
        self.backbone = LightweightBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
        )
        
        # Parametric Temporal Correlation Fusion (~280K params)
        self.ptcf = PTCF(
            in_channels=backbone_out,
            out_channels=ptcf_out_channels,
        )
        
        # Adaptive Temporal Similarity Matrices (~150K params)
        self.atsm = ATSM(
            in_channels=ptcf_out_channels,
            num_classes=num_classes,
        )
        
        # Meta-Learned Change Boundaries (~80K params)
        self.mcb = MCB(
            in_channels=ptcf_out_channels,
            num_classes=num_classes,
        )
        
        # Decoder (~80K params)
        self.decoder = SimpleDecoder(
            in_channels=ptcf_out_channels,
            hidden_channels=decoder_channels,
            num_classes=1,
        )
        
        # Store config
        self.config = {
            "in_channels": in_channels,
            "backbone_channels": backbone_channels,
            "ptcf_out_channels": ptcf_out_channels,
            "num_classes": num_classes,
            "decoder_channels": decoder_channels,
        }
    
    def extract_features(
        self,
        img_t1: torch.Tensor,
        img_t2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract backbone features from bi-temporal images.
        
        Args:
            img_t1: Image at time T₁ (B, C, H, W)
            img_t2: Image at time T₂ (B, C, H, W)
        
        Returns:
            Tuple of feature tensors (F_T₁, F_T₂)
        """
        f_t1 = self.backbone(img_t1)
        f_t2 = self.backbone(img_t2)
        return f_t1, f_t2
    
    def forward(
        self,
        img_t1: torch.Tensor,
        img_t2: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
        support_masks: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for change detection.
        
        Args:
            img_t1: Image at time T₁ (B, 3, H, W)
            img_t2: Image at time T₂ (B, 3, H, W)
            support_features: Optional support set features for meta-learning
            support_masks: Optional support set masks
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary containing:
                - 'pred': Change probability map (B, 1, H, W)
                - 'thresholds': Learned thresholds (B, num_classes)
                - 'uncertainty': Uncertainty scores (B, 1)
                - 'features': (optional) Intermediate features
        """
        # Step 1: Extract backbone features
        f_t1, f_t2 = self.extract_features(img_t1, img_t2)
        
        # Step 2: Temporal correlation fusion
        z = self.ptcf(f_t1, f_t2)  # (B, 256, H/8, W/8)
        
        # Step 3: Adaptive temporal similarity
        z = self.atsm(z, support_features, support_masks)  # (B, 256, H/8, W/8)
        
        # Step 4: Meta-learned change boundaries
        refined, thresholds, uncertainty = self.mcb(
            z, support_features, support_masks
        )
        
        # Step 5: Decode to prediction mask
        pred = self.decoder(z)  # (B, 1, H, W)
        
        outputs = {
            "pred": pred,
            "thresholds": thresholds,
            "uncertainty": uncertainty,
        }
        
        if return_features:
            outputs["features"] = {
                "f_t1": f_t1,
                "f_t2": f_t2,
                "z": z,
                "refined": refined,
            }
        
        return outputs
    
    def predict(
        self,
        img_t1: torch.Tensor,
        img_t2: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate binary change prediction.
        
        Args:
            img_t1: Image at time T₁ (B, 3, H, W)
            img_t2: Image at time T₂ (B, 3, H, W)
            threshold: Optional fixed threshold (uses learned if None)
        
        Returns:
            Binary change mask (B, 1, H, W)
        """
        with torch.no_grad():
            outputs = self.forward(img_t1, img_t2)
            
            if threshold is not None:
                binary_pred = (outputs["pred"] > threshold).float()
            else:
                binary_pred = self.mcb.apply_thresholds(
                    outputs["pred"],
                    outputs["thresholds"],
                )
        
        return binary_pred
    
    def get_num_params(self, by_module: bool = False) -> Any:
        """Get number of parameters.
        
        Args:
            by_module: If True, return dict with per-module counts
        
        Returns:
            Total parameter count or dict with per-module counts
        """
        if by_module:
            return {
                "backbone": self.backbone.get_num_params(),
                "ptcf": self.ptcf.get_num_params(),
                "atsm": self.atsm.get_num_params(),
                "mcb": self.mcb.get_num_params(),
                "decoder": self.decoder.get_num_params(),
                "total": sum(p.numel() for p in self.parameters()),
            }
        return sum(p.numel() for p in self.parameters())
    
    def freeze_backbone(self) -> None:
        """Freeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_meta_parameters(self) -> list:
        """Get parameters that should be meta-learned.
        
        Returns list of parameter groups for meta-optimization.
        """
        # Temporal correlation weights
        meta_params = [self.ptcf.W_temporal]
        
        # ATSM temperature
        meta_params.append(self.atsm.temperature)
        
        # MCB base thresholds
        meta_params.append(self.mcb.base_thresholds)
        
        return meta_params


def create_model(config=None) -> TemporalCorrMetaNet:
    """Factory function to create model from config.
    
    Args:
        config: ModelConfig object or None for defaults
    
    Returns:
        TemporalCorrMetaNet instance
    """
    if config is None:
        return TemporalCorrMetaNet()
    
    return TemporalCorrMetaNet(
        in_channels=config.in_channels,
        backbone_channels=config.backbone_channels,
        ptcf_out_channels=config.ptcf_out_channels,
        num_classes=config.num_classes,
        decoder_channels=config.decoder_channels,
    )


if __name__ == "__main__":
    # Test complete model
    model = TemporalCorrMetaNet()
    
    # Print parameter counts
    param_counts = model.get_num_params(by_module=True)
    print("Parameter counts by module:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    img_t1 = torch.randn(2, 3, 512, 512)
    img_t2 = torch.randn(2, 3, 512, 512)
    
    outputs = model(img_t1, img_t2)
    
    print(f"\nInput shapes: T1={img_t1.shape}, T2={img_t2.shape}")
    print(f"Prediction shape: {outputs['pred'].shape}")
    print(f"Thresholds shape: {outputs['thresholds'].shape}")
    print(f"Uncertainty shape: {outputs['uncertainty'].shape}")
    print(f"Pred range: [{outputs['pred'].min():.3f}, {outputs['pred'].max():.3f}]")
    
    # Test prediction
    binary_pred = model.predict(img_t1, img_t2)
    print(f"Binary prediction shape: {binary_pred.shape}")
    print(f"Unique values: {binary_pred.unique().tolist()}")
