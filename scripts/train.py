"""Training script for TemporalCorr-MetaNet.

Implements episodic meta-training from Section 6 of technical document.

Usage:
    python scripts/train.py --config configs/experiment_configs/egybcd.yaml
    python scripts/train.py --dataset EGY-BCD --data-root ./data/EGY-BCD
"""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import Config, get_default_config
from models import TemporalCorrMetaNet, MetaLearner
from data import (
    EGYBCDDataset,
    WHUDataset,
    LEVIRCDDataset,
    S2LookingDataset,
    EpisodeSampler,
    get_train_transforms,
    get_val_transforms,
)
from utils import MetaLearningLoss, MetricTracker, plot_training_curves


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train_one_episode(
    meta_learner: MetaLearner,
    episode,
    loss_fn,
    device: torch.device,
):
    """Train on a single episode."""
    # Move data to device
    episode = episode.to(device)
    
    # Meta-training step
    metrics = meta_learner.episode_train(
        episode.support_t1,
        episode.support_t2,
        episode.support_masks,
        episode.query_t1,
        episode.query_t2,
        episode.query_masks,
        loss_fn,
    )
    
    return metrics


def validate(
    meta_learner: MetaLearner,
    val_sampler: EpisodeSampler,
    loss_fn,
    device: torch.device,
    num_episodes: int = 100,
):
    """Validate on multiple episodes."""
    tracker = MetricTracker()
    
    for episode in tqdm(val_sampler, desc="Validating", total=num_episodes, leave=False):
        episode = episode.to(device)
        
        metrics = meta_learner.evaluate_episode(
            episode.support_t1,
            episode.support_t2,
            episode.support_masks,
            episode.query_t1,
            episode.query_t2,
            episode.query_masks,
            loss_fn,
        )
        
        # Update tracker
        with torch.no_grad():
            outputs = meta_learner.model(episode.query_t1, episode.query_t2)
            tracker.update(outputs["pred"], episode.query_masks.unsqueeze(1))
    
    return tracker.compute()


def main(args=None):
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train TemporalCorr-MetaNet")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="EGY-BCD", 
                       help="Dataset name (EGY-BCD, WHU, LEVIR-CD, S2Looking)")
    parser.add_argument("--data-root", type=str, help="Dataset root directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--max-episodes", type=int, default=10000)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--num-support", type=int, default=5, help="K-shot")
    parser.add_argument("--num-query", type=int, default=15)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--outer-lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args(args)
    
    # Load config
    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = get_default_config()
    
    # Override with command line args
    if args.data_root:
        config.data.data_root = args.data_root
    config.training.seed = args.seed
    config.training.device = args.device
    config.meta.max_episodes = args.max_episodes
    config.meta.val_interval = args.val_interval
    config.meta.support_samples = args.num_support
    config.meta.query_samples = args.num_query
    config.meta.inner_lr = args.inner_lr
    config.meta.outer_lr = args.outer_lr
    
    # Set seed
    set_seed(config.training.seed)
    
    # Device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_{config.meta.support_samples}shot_{timestamp}"
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name
    log_dir = Path(args.log_dir) / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(str(checkpoint_dir / "config.yaml"))
    
    # Transforms
    train_transform = get_train_transforms(
        image_size=config.data.image_size,
        rotation_range=config.data.rotation_range,
        flip_prob=config.data.flip_prob,
        brightness_range=config.data.brightness_range,
        contrast_range=config.data.contrast_range,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )
    val_transform = get_val_transforms(
        image_size=config.data.image_size,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )
    
    # Datasets
    print(f"Loading dataset: {args.dataset} from {config.data.data_root}")
    train_dataset = get_dataset(args.dataset, config.data.data_root, "train", train_transform)
    val_dataset = get_dataset(args.dataset, config.data.data_root, "val", val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Episode samplers
    train_sampler = EpisodeSampler(
        train_dataset,
        num_support=config.meta.support_samples,
        num_query=config.meta.query_samples,
        num_episodes=config.meta.max_episodes,
        seed=config.training.seed,
    )
    val_sampler = EpisodeSampler(
        val_dataset,
        num_support=config.meta.support_samples,
        num_query=config.meta.query_samples,
        num_episodes=100,
        seed=config.training.seed + 1,
    )
    
    # Model
    model = TemporalCorrMetaNet(
        in_channels=config.model.in_channels,
        backbone_channels=config.model.backbone_channels,
        ptcf_out_channels=config.model.ptcf_out_channels,
        num_classes=config.model.num_classes,
        decoder_channels=config.model.decoder_channels,
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    param_counts = model.get_num_params(by_module=True)
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Meta-learner
    meta_learner = MetaLearner(
        model,
        inner_lr=config.meta.inner_lr,
        outer_lr=config.meta.outer_lr,
        inner_steps=config.meta.inner_loop_steps,
        weight_decay=config.meta.weight_decay,
        gradient_clip=config.meta.gradient_clip,
    )
    
    # Loss function
    loss_fn = MetaLearningLoss(
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        sparsity_weight=config.loss.correlation_sparsity_weight,
        smoothness_weight=config.loss.boundary_smoothness_weight,
        uncertainty_weight=config.loss.uncertainty_reg_weight,
    )
    
    # Resume if specified
    start_episode = 0
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        checkpoint = meta_learner.load_checkpoint(args.resume)
        start_episode = checkpoint.get("episode_count", 0)
    
    # Tensorboard
    writer = SummaryWriter(log_dir)
    
    # Training history
    history = {
        "train_loss": [],
        "train_iou": [],
        "val_iou": [],
        "val_f1": [],
    }
    
    # Training loop
    best_val_iou = 0.0
    patience_counter = 0
    
    print(f"\nStarting training from episode {start_episode}")
    print(f"Max episodes: {config.meta.max_episodes}")
    print(f"K-shot: {config.meta.support_samples}, N-query: {config.meta.query_samples}")
    print("-" * 50)
    
    pbar = tqdm(train_sampler, desc="Training", total=config.meta.max_episodes)
    for episode_idx, episode in enumerate(pbar, start=start_episode):
        if episode_idx >= config.meta.max_episodes:
            break
        
        # Train step
        metrics = train_one_episode(
            meta_learner,
            episode,
            lambda pred, target: loss_fn({"pred": pred, "uncertainty": torch.rand(pred.shape[0], 1, device=pred.device)}, target)["total"],
            device,
        )
        
        # Update history
        history["train_loss"].append(metrics["total_loss"])
        history["train_iou"].append(metrics["query_iou"])
        
        # Logging
        writer.add_scalar("train/loss", metrics["total_loss"], episode_idx)
        writer.add_scalar("train/support_loss", metrics["support_loss"], episode_idx)
        writer.add_scalar("train/query_loss", metrics["query_loss"], episode_idx)
        writer.add_scalar("train/iou", metrics["query_iou"], episode_idx)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{metrics['total_loss']:.4f}",
            "iou": f"{metrics['query_iou']:.4f}",
        })
        
        # Validation
        if (episode_idx + 1) % config.meta.val_interval == 0:
            val_metrics = validate(
                meta_learner,
                val_sampler,
                lambda pred, target: loss_fn({"pred": pred, "uncertainty": torch.rand(pred.shape[0], 1, device=pred.device)}, target)["total"],
                device,
            )
            
            val_iou = val_metrics.get("iou_global", val_metrics.get("iou_avg", 0))
            val_f1 = val_metrics.get("f1_global", val_metrics.get("f1_avg", 0))
            
            history["val_iou"].append(val_iou)
            history["val_f1"].append(val_f1)
            
            writer.add_scalar("val/iou", val_iou, episode_idx)
            writer.add_scalar("val/f1", val_f1, episode_idx)
            
            print(f"\n[Episode {episode_idx + 1}] Val IoU: {val_iou:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                patience_counter = 0
                meta_learner.save_checkpoint(
                    str(checkpoint_dir / "best_model.pth"),
                    {"val_iou": val_iou, "val_f1": val_f1},
                )
                print(f"  New best model! IoU: {val_iou:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config.training.patience:
                print(f"\nEarly stopping at episode {episode_idx + 1}")
                break
        
        # Save checkpoint
        if (episode_idx + 1) % args.save_interval == 0:
            meta_learner.save_checkpoint(
                str(checkpoint_dir / f"checkpoint_ep{episode_idx + 1}.pth"),
            )
    
    # Save final model
    meta_learner.save_checkpoint(str(checkpoint_dir / "final_model.pth"))
    
    # Save training curves
    fig = plot_training_curves(history, str(checkpoint_dir / "training_curves.png"))
    
    # Close tensorboard
    writer.close()
    
    print("\nTraining completed!")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    main()
