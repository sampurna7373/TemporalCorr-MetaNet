"""Evaluation script for TemporalCorr-MetaNet.

Evaluates trained model on test set with comprehensive metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --dataset EGY-BCD
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import Config
from models import TemporalCorrMetaNet
from data import (
    EGYBCDDataset,
    WHUDataset,
    LEVIRCDDataset,
    S2LookingDataset,
    EpisodeSampler,
    get_val_transforms,
)
from utils import (
    MetricTracker,
    compute_all_metrics,
    visualize_prediction,
)


def get_dataset(name: str, root: str, split: str, transform):
    """Get dataset by name."""
    name = name.lower().replace("-", "").replace("_", "")
    
    if name in ["egybcd", "egypt"]:
        return EGYBCDDataset(root, split, transform)
    elif name in ["whu", "whubcd"]:
        return WHUDataset(root, split, transform)
    elif name in ["levircd", "levir"]:
        return LEVIRCDDataset(root, split, transform)
    elif name in ["s2looking", "s2"]:
        return S2LookingDataset(root, split, transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def evaluate_standard(
    model: TemporalCorrMetaNet,
    dataset,
    device: torch.device,
    batch_size: int = 4,
    threshold: float = 0.5,
    save_vis: bool = False,
    vis_dir: str = None,
):
    """Standard evaluation on full test set."""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    tracker = MetricTracker()
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            img_t1 = batch["img_t1"].to(device)
            img_t2 = batch["img_t2"].to(device)
            masks = batch["mask"].to(device)
            
            outputs = model(img_t1, img_t2)
            pred = outputs["pred"]
            
            tracker.update(pred, masks.unsqueeze(1), threshold)
            
            # Save visualizations
            if save_vis and vis_dir and batch_idx < 10:
                for i in range(min(2, img_t1.shape[0])):
                    visualize_prediction(
                        img_t1[i],
                        img_t2[i],
                        pred[i],
                        masks[i],
                        threshold=threshold,
                        save_path=os.path.join(vis_dir, f"vis_{batch_idx}_{i}.png"),
                    )
    
    return tracker.compute()


def evaluate_fewshot(
    model: TemporalCorrMetaNet,
    dataset,
    device: torch.device,
    num_support: int = 5,
    num_episodes: int = 100,
    threshold: float = 0.5,
):
    """Few-shot evaluation using episodic protocol."""
    sampler = EpisodeSampler(
        dataset,
        num_support=num_support,
        num_query=15,
        num_episodes=num_episodes,
    )
    
    tracker = MetricTracker()
    model.eval()
    
    with torch.no_grad():
        for episode in tqdm(sampler, desc=f"Evaluating {num_support}-shot"):
            episode = episode.to(device)
            
            # Get support features
            support_outputs = model(
                episode.support_t1,
                episode.support_t2,
                return_features=True,
            )
            support_features = support_outputs["features"]["z"]
            
            # Predict on query set
            query_outputs = model(
                episode.query_t1,
                episode.query_t2,
                support_features=support_features,
                support_masks=episode.support_masks,
            )
            
            tracker.update(query_outputs["pred"], episode.query_masks.unsqueeze(1), threshold)
    
    return tracker.compute()


def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate TemporalCorr-MetaNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, help="Path to config (auto-detected from checkpoint)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data-root", type=str, help="Dataset root")
    parser.add_argument("--split", type=str, default="test", help="Data split")
    parser.add_argument("--mode", type=str, default="both", 
                       choices=["standard", "fewshot", "both"])
    parser.add_argument("--num-support", type=int, nargs="+", default=[1, 5, 10],
                       help="K-shot values for few-shot evaluation")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--output", type=str, default="./results")
    
    args = parser.parse_args(args)
    
    # Load config
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = args.config or checkpoint_dir / "config.yaml"
    
    if config_path.exists():
        config = Config.load(str(config_path))
    else:
        config = Config()
    
    # Override data root
    if args.data_root:
        config.data.data_root = args.data_root
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = TemporalCorrMetaNet(
        in_channels=config.model.in_channels,
        backbone_channels=config.model.backbone_channels,
        ptcf_out_channels=config.model.ptcf_out_channels,
        num_classes=config.model.num_classes,
        decoder_channels=config.model.decoder_channels,
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Dataset
    transform = get_val_transforms(
        image_size=config.data.image_size,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )
    
    print(f"Loading {args.dataset} ({args.split} split) from {config.data.data_root}")
    dataset = get_dataset(args.dataset, config.data.data_root, args.split, transform)
    print(f"Test samples: {len(dataset)}")
    
    # Output directory
    output_dir = Path(args.output) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Standard evaluation
    if args.mode in ["standard", "both"]:
        print("\n" + "=" * 50)
        print("Standard Evaluation")
        print("=" * 50)
        
        vis_dir = output_dir / "visualizations" if args.save_vis else None
        if vis_dir:
            vis_dir.mkdir(parents=True, exist_ok=True)
        
        standard_metrics = evaluate_standard(
            model,
            dataset,
            device,
            threshold=args.threshold,
            save_vis=args.save_vis,
            vis_dir=str(vis_dir) if vis_dir else None,
        )
        
        results["standard"] = standard_metrics
        
        print("\nStandard Evaluation Results:")
        print(f"  IoU: {standard_metrics['iou_global']:.4f}")
        print(f"  F1: {standard_metrics['f1_global']:.4f}")
        print(f"  Precision: {standard_metrics['precision_global']:.4f}")
        print(f"  Recall: {standard_metrics['recall_global']:.4f}")
        print(f"  Accuracy: {standard_metrics['accuracy_global']:.4f}")
    
    # Few-shot evaluation
    if args.mode in ["fewshot", "both"]:
        print("\n" + "=" * 50)
        print("Few-Shot Evaluation")
        print("=" * 50)
        
        results["fewshot"] = {}
        
        for k in args.num_support:
            print(f"\n{k}-shot evaluation:")
            
            fewshot_metrics = evaluate_fewshot(
                model,
                dataset,
                device,
                num_support=k,
                num_episodes=args.num_episodes,
                threshold=args.threshold,
            )
            
            results["fewshot"][f"{k}-shot"] = fewshot_metrics
            
            print(f"  IoU: {fewshot_metrics['iou_global']:.4f}")
            print(f"  F1: {fewshot_metrics['f1_global']:.4f}")
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
