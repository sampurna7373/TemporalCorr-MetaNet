"""Visualization utilities for change detection.

Provides functions for:
- Visualizing predictions vs ground truth
- Plotting training curves
- Visualizing few-shot episodes
- Correlation weight visualization
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first sample
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    return tensor.detach().cpu().numpy()


def denormalize_image(
    img: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Denormalize image for visualization."""
    img = img.copy()
    for i in range(3):
        img[..., i] = img[..., i] * std[i] + mean[i]
    return np.clip(img, 0, 1)


def visualize_prediction(
    img_t1: torch.Tensor,
    img_t2: torch.Tensor,
    pred: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
) -> Figure:
    """Visualize change detection prediction.
    
    Args:
        img_t1: Image at T1 (1, 3, H, W) or (3, H, W)
        img_t2: Image at T2 (1, 3, H, W) or (3, H, W)
        pred: Prediction (1, 1, H, W) or (1, H, W)
        target: Optional ground truth
        threshold: Threshold for binarization
        save_path: Optional path to save figure
        title: Optional title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Convert tensors
    t1 = tensor_to_numpy(img_t1)
    t2 = tensor_to_numpy(img_t2)
    p = tensor_to_numpy(pred)
    if p.ndim == 3 and p.shape[-1] == 1:
        p = p.squeeze(-1)
    
    if target is not None:
        gt = tensor_to_numpy(target)
        if gt.ndim == 3 and gt.shape[-1] == 1:
            gt = gt.squeeze(-1)
    else:
        gt = None
    
    # Denormalize images
    t1 = denormalize_image(t1)
    t2 = denormalize_image(t2)
    
    # Create figure
    n_cols = 4 if gt is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    axes[0].imshow(t1)
    axes[0].set_title("T1 (Before)")
    axes[0].axis("off")
    
    axes[1].imshow(t2)
    axes[1].set_title("T2 (After)")
    axes[1].axis("off")
    
    # Prediction
    pred_binary = (p > threshold).astype(np.float32)
    axes[2].imshow(pred_binary, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    
    if gt is not None:
        # Create difference map
        diff = np.zeros((*gt.shape, 3))
        # True positive: green
        diff[(pred_binary == 1) & (gt == 1)] = [0, 1, 0]
        # False positive: red
        diff[(pred_binary == 1) & (gt == 0)] = [1, 0, 0]
        # False negative: blue
        diff[(pred_binary == 0) & (gt == 1)] = [0, 0, 1]
        
        axes[3].imshow(diff)
        axes[3].set_title("Comparison (G=TP, R=FP, B=FN)")
        axes[3].axis("off")
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def visualize_episode(
    episode: Any,  # FewShotBatch
    predictions: Optional[torch.Tensor] = None,
    num_show: int = 4,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12),
) -> Figure:
    """Visualize a few-shot episode.
    
    Args:
        episode: FewShotBatch object
        predictions: Optional predictions for query set
        num_show: Number of samples to show per set
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_support = min(num_show, episode.support_t1.shape[0])
    n_query = min(num_show, episode.query_t1.shape[0])
    
    fig = plt.figure(figsize=figsize)
    
    # Support set (top rows)
    for i in range(n_support):
        # T1
        ax = fig.add_subplot(4, num_show, i + 1)
        img = tensor_to_numpy(episode.support_t1[i])
        ax.imshow(denormalize_image(img))
        ax.set_title(f"Support T1 [{i}]" if i == 0 else f"[{i}]")
        ax.axis("off")
        
        # Mask
        ax = fig.add_subplot(4, num_show, num_show + i + 1)
        mask = tensor_to_numpy(episode.support_masks[i])
        ax.imshow(mask, cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_title(f"Support Mask" if i == 0 else "")
        ax.axis("off")
    
    # Query set (bottom rows)
    for i in range(n_query):
        # T1/T2
        ax = fig.add_subplot(4, num_show, 2 * num_show + i + 1)
        img = tensor_to_numpy(episode.query_t1[i])
        ax.imshow(denormalize_image(img))
        ax.set_title(f"Query T1 [{i}]" if i == 0 else f"[{i}]")
        ax.axis("off")
        
        # Mask or Prediction
        ax = fig.add_subplot(4, num_show, 3 * num_show + i + 1)
        if predictions is not None:
            pred = tensor_to_numpy(predictions[i])
            if pred.ndim > 2:
                pred = pred.squeeze()
            ax.imshow(pred, cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_title(f"Prediction" if i == 0 else "")
        else:
            mask = tensor_to_numpy(episode.query_masks[i])
            ax.imshow(mask, cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_title(f"Query Mask" if i == 0 else "")
        ax.axis("off")
    
    plt.suptitle("Few-Shot Episode", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> Figure:
    """Plot training metrics curves.
    
    Args:
        metrics: Dictionary with metric name -> list of values
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Organize metrics into groups
    loss_metrics = {k: v for k, v in metrics.items() if "loss" in k.lower()}
    perf_metrics = {k: v for k, v in metrics.items() if k.lower() in 
                   ["iou", "miou", "f1", "precision", "recall", "accuracy"]}
    other_metrics = {k: v for k, v in metrics.items() 
                    if k not in loss_metrics and k not in perf_metrics}
    
    n_plots = sum(1 for m in [loss_metrics, perf_metrics, other_metrics] if m)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if loss_metrics:
        ax = axes[plot_idx]
        for name, values in loss_metrics.items():
            ax.plot(values, label=name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    if perf_metrics:
        ax = axes[plot_idx]
        for name, values in perf_metrics.items():
            ax.plot(values, label=name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.set_title("Performance Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plot_idx += 1
    
    if other_metrics:
        ax = axes[plot_idx]
        for name, values in other_metrics.items():
            ax.plot(values, label=name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.set_title("Other Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def visualize_correlation_weights(
    W_temporal: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """Visualize learned temporal correlation weights.
    
    Args:
        W_temporal: Correlation weight matrix (C, C)
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    W = W_temporal.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Raw weights
    im = axes[0].imshow(W, cmap="RdBu", vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    axes[0].set_title("Temporal Correlation Weights")
    axes[0].set_xlabel("Channel (T2)")
    axes[0].set_ylabel("Channel (T1)")
    plt.colorbar(im, ax=axes[0])
    
    # Absolute weights (importance)
    W_abs = np.abs(W)
    im = axes[1].imshow(W_abs, cmap="hot")
    axes[1].set_title("Weight Magnitude (Importance)")
    axes[1].set_xlabel("Channel (T2)")
    axes[1].set_ylabel("Channel (T1)")
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def create_comparison_figure(
    methods: Dict[str, torch.Tensor],
    target: torch.Tensor,
    img_t1: torch.Tensor,
    img_t2: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> Figure:
    """Create comparison figure with multiple methods.
    
    Args:
        methods: Dictionary mapping method name to predictions
        target: Ground truth
        img_t1, img_t2: Input images
        save_path: Path to save
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_methods = len(methods)
    figsize = figsize or (4 * (n_methods + 3), 4)
    
    fig, axes = plt.subplots(1, n_methods + 3, figsize=figsize)
    
    # T1
    t1 = tensor_to_numpy(img_t1)
    axes[0].imshow(denormalize_image(t1))
    axes[0].set_title("T1")
    axes[0].axis("off")
    
    # T2
    t2 = tensor_to_numpy(img_t2)
    axes[1].imshow(denormalize_image(t2))
    axes[1].set_title("T2")
    axes[1].axis("off")
    
    # Ground truth
    gt = tensor_to_numpy(target)
    if gt.ndim > 2:
        gt = gt.squeeze()
    axes[2].imshow(gt, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")
    
    # Methods
    for i, (name, pred) in enumerate(methods.items()):
        p = tensor_to_numpy(pred)
        if p.ndim > 2:
            p = p.squeeze()
        p_binary = (p > 0.5).astype(np.float32)
        axes[3 + i].imshow(p_binary, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[3 + i].set_title(name)
        axes[3 + i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


if __name__ == "__main__":
    # Test visualizations
    img_t1 = torch.randn(1, 3, 256, 256)
    img_t2 = torch.randn(1, 3, 256, 256)
    pred = torch.sigmoid(torch.randn(1, 1, 256, 256))
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    fig = visualize_prediction(img_t1, img_t2, pred, target, save_path=None)
    plt.close(fig)
    print("visualize_prediction: OK")
    
    # Test training curves
    metrics = {
        "loss": [1.0, 0.8, 0.6, 0.5, 0.4],
        "iou": [0.3, 0.4, 0.5, 0.55, 0.6],
        "f1": [0.4, 0.5, 0.6, 0.65, 0.7],
    }
    fig = plot_training_curves(metrics, save_path=None)
    plt.close(fig)
    print("plot_training_curves: OK")
    
    # Test correlation weights
    W = torch.randn(64, 64)
    fig = visualize_correlation_weights(W, save_path=None)
    plt.close(fig)
    print("visualize_correlation_weights: OK")
