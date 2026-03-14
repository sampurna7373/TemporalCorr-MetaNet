"""Ablation study script for TemporalCorr-MetaNet.

Runs ablation experiments from Section 6.4 of technical document:
- Without PTCF (simple concatenation)
- Without ATSM (no class-specific similarity)
- Without MCB (fixed threshold)
- Without meta-learning (standard training)

Usage:
    python scripts/ablation_study.py --dataset EGY-BCD --data-root ./data/EGY-BCD
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import get_default_config
from models import TemporalCorrMetaNet, LightweightBackbone
from models.decoder import SimpleDecoder
from data import EGYBCDDataset, EpisodeSampler, get_train_transforms, get_val_transforms
from utils import MetricTracker, FocalLoss


class BaselineConcat(nn.Module):
    """Baseline: Simple feature concatenation instead of PTCF."""
    
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.backbone = LightweightBackbone(in_channels)
        backbone_out = 64
        
        # Simple concatenation + projection
        self.fusion = nn.Sequential(
            nn.Conv2d(backbone_out * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = SimpleDecoder(out_channels)
    
    def forward(self, img_t1, img_t2, **kwargs):
        f_t1 = self.backbone(img_t1)
        f_t2 = self.backbone(img_t2)
        
        z = torch.cat([f_t1, f_t2], dim=1)
        z = self.fusion(z)
        pred = self.decoder(z)
        
        return {"pred": pred, "thresholds": None, "uncertainty": None}
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class BaselineDiff(nn.Module):
    """Baseline: Feature difference instead of PTCF."""
    
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.backbone = LightweightBackbone(in_channels)
        backbone_out = 64
        
        # Difference + projection
        self.fusion = nn.Sequential(
            nn.Conv2d(backbone_out, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = SimpleDecoder(out_channels)
    
    def forward(self, img_t1, img_t2, **kwargs):
        f_t1 = self.backbone(img_t1)
        f_t2 = self.backbone(img_t2)
        
        diff = torch.abs(f_t2 - f_t1)
        z = self.fusion(diff)
        pred = self.decoder(z)
        
        return {"pred": pred, "thresholds": None, "uncertainty": None}
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class NoPTCF(TemporalCorrMetaNet):
    """Ablation: Without PTCF module (use simple concatenation)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        backbone_out = 64
        
        # Replace PTCF with simple fusion
        self.ptcf = nn.Sequential(
            nn.Conv2d(backbone_out * 2, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, img_t1, img_t2, **kwargs):
        f_t1, f_t2 = self.extract_features(img_t1, img_t2)
        
        # Simple concatenation instead of PTCF
        z = torch.cat([f_t1, f_t2], dim=1)
        z = self.ptcf(z)
        
        z = self.atsm(z)
        refined, thresholds, uncertainty = self.mcb(z)
        pred = self.decoder(z)
        
        return {"pred": pred, "thresholds": thresholds, "uncertainty": uncertainty}


class NoATSM(TemporalCorrMetaNet):
    """Ablation: Without ATSM module."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace ATSM with identity
        self.atsm = nn.Identity()
    
    def forward(self, img_t1, img_t2, **kwargs):
        f_t1, f_t2 = self.extract_features(img_t1, img_t2)
        z = self.ptcf(f_t1, f_t2)
        z = self.atsm(z)  # Identity
        refined, thresholds, uncertainty = self.mcb(z)
        pred = self.decoder(z)
        
        return {"pred": pred, "thresholds": thresholds, "uncertainty": uncertainty}


class NoMCB(TemporalCorrMetaNet):
    """Ablation: Without MCB module (fixed threshold)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Remove MCB
        self.mcb = None
    
    def forward(self, img_t1, img_t2, **kwargs):
        f_t1, f_t2 = self.extract_features(img_t1, img_t2)
        z = self.ptcf(f_t1, f_t2)
        z = self.atsm(z)
        pred = self.decoder(z)
        
        # Fixed threshold
        return {
            "pred": pred,
            "thresholds": torch.tensor([[0.5, 0.5]], device=pred.device).expand(pred.shape[0], -1),
            "uncertainty": torch.zeros(pred.shape[0], 1, device=pred.device),
        }


def train_model(
    model,
    train_sampler,
    val_sampler,
    device,
    num_episodes=2000,
    lr=0.001,
):
    """Train a model for ablation study."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss()
    
    best_val_iou = 0
    
    for episode_idx, episode in enumerate(tqdm(train_sampler, total=num_episodes)):
        if episode_idx >= num_episodes:
            break
        
        model.train()
        episode = episode.to(device)
        
        # Forward pass on query
        outputs = model(episode.query_t1, episode.query_t2)
        loss = loss_fn(outputs["pred"], episode.query_masks.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        if (episode_idx + 1) % 500 == 0:
            model.eval()
            tracker = MetricTracker()
            
            with torch.no_grad():
                for val_episode in val_sampler:
                    val_episode = val_episode.to(device)
                    outputs = model(val_episode.query_t1, val_episode.query_t2)
                    tracker.update(outputs["pred"], val_episode.query_masks.unsqueeze(1))
            
            metrics = tracker.compute()
            val_iou = metrics.get("iou_global", 0)
            
            if val_iou > best_val_iou:
                best_val_iou = val_iou
    
    return best_val_iou


def main(args=None):
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--dataset", type=str, default="EGY-BCD")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="./results/ablation")
    
    args = parser.parse_args(args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    config = get_default_config()
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    from data import EGYBCDDataset as Dataset
    train_dataset = Dataset(args.data_root, "train", train_transform)
    val_dataset = Dataset(args.data_root, "val", val_transform)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Models to evaluate
    models = {
        "Full Model": TemporalCorrMetaNet,
        "No PTCF (Concat)": NoPTCF,
        "No ATSM": NoATSM,
        "No MCB": NoMCB,
        "Baseline (Concat)": BaselineConcat,
        "Baseline (Diff)": BaselineDiff,
    }
    
    results = {}
    
    for name, ModelClass in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        model = ModelClass().to(device)
        print(f"Parameters: {model.get_num_params():,}")
        
        train_sampler = EpisodeSampler(train_dataset, num_episodes=args.num_episodes)
        val_sampler = EpisodeSampler(val_dataset, num_episodes=50)
        
        best_iou = train_model(
            model,
            train_sampler,
            val_sampler,
            device,
            num_episodes=args.num_episodes,
        )
        
        results[name] = {
            "best_val_iou": best_iou,
            "params": model.get_num_params(),
        }
        
        print(f"Best Val IoU: {best_iou:.4f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'Val IoU':<12} {'Params':<12}")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<25} {data['best_val_iou']:.4f}       {data['params']:,}")
    
    return results


if __name__ == "__main__":
    main()
