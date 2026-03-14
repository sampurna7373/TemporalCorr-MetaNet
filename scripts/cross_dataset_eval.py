"""Cross-dataset generalization evaluation.

Evaluates model trained on one dataset and tested on others.
From Section 6.5 of technical document.

Usage:
    python scripts/cross_dataset_eval.py --checkpoint checkpoints/egybcd/best_model.pth
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
from utils import MetricTracker


DATASETS = {
    "EGY-BCD": EGYBCDDataset,
    "WHU": WHUDataset,
    "LEVIR-CD": LEVIRCDDataset,
    "S2Looking": S2LookingDataset,
}


def evaluate_on_dataset(model, dataset_class, data_root, device, num_episodes=100):
    """Evaluate model on a specific dataset."""
    transform = get_val_transforms()
    
    try:
        dataset = dataset_class(data_root, "test", transform)
    except Exception as e:
        print(f"  Could not load dataset: {e}")
        return None
    
    if len(dataset) == 0:
        print(f"  Dataset is empty")
        return None
    
    sampler = EpisodeSampler(dataset, num_episodes=num_episodes)
    tracker = MetricTracker()
    
    model.eval()
    with torch.no_grad():
        for episode in tqdm(sampler, desc="Evaluating", leave=False):
            episode = episode.to(device)
            
            # Get support features
            support_outputs = model(
                episode.support_t1,
                episode.support_t2,
                return_features=True,
            )
            support_features = support_outputs["features"]["z"]
            
            # Predict on query
            query_outputs = model(
                episode.query_t1,
                episode.query_t2,
                support_features=support_features,
                support_masks=episode.support_masks,
            )
            
            tracker.update(query_outputs["pred"], episode.query_masks.unsqueeze(1))
    
    return tracker.compute()


def main(args=None):
    parser = argparse.ArgumentParser(description="Cross-Dataset Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source-dataset", type=str, default="EGY-BCD",
                       help="Dataset the model was trained on")
    parser.add_argument("--data-roots", type=str, nargs="+", required=True,
                       help="Data roots for each dataset (same order as DATASETS)")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="./results/cross_dataset")
    
    args = parser.parse_args(args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = TemporalCorrMetaNet()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Build data root mapping
    dataset_names = list(DATASETS.keys())
    if len(args.data_roots) != len(dataset_names):
        print(f"Warning: Expected {len(dataset_names)} data roots, got {len(args.data_roots)}")
        # Pad with None
        args.data_roots = args.data_roots + [None] * (len(dataset_names) - len(args.data_roots))
    
    data_roots = dict(zip(dataset_names, args.data_roots))
    
    # Evaluate on all datasets
    results = {}
    
    print(f"\nSource dataset: {args.source_dataset}")
    print("=" * 60)
    
    for name, dataset_class in DATASETS.items():
        print(f"\nEvaluating on: {name}")
        
        root = data_roots.get(name)
        if not root or not Path(root).exists():
            print(f"  Skipping (no data root)")
            continue
        
        metrics = evaluate_on_dataset(
            model,
            dataset_class,
            root,
            device,
            args.num_episodes,
        )
        
        if metrics:
            results[name] = {
                "iou": metrics["iou_global"],
                "f1": metrics["f1_global"],
                "precision": metrics["precision_global"],
                "recall": metrics["recall_global"],
            }
            print(f"  IoU: {metrics['iou_global']:.4f}")
            print(f"  F1: {metrics['f1_global']:.4f}")
    
    # Compute generalization drop
    if args.source_dataset in results:
        source_iou = results[args.source_dataset]["iou"]
        print("\n" + "=" * 60)
        print("GENERALIZATION ANALYSIS")
        print("=" * 60)
        
        for name, data in results.items():
            drop = source_iou - data["iou"]
            drop_pct = (drop / source_iou) * 100 if source_iou > 0 else 0
            print(f"{name}: IoU={data['iou']:.4f}, Drop={drop_pct:+.1f}%")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"cross_dataset_{args.source_dataset}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
