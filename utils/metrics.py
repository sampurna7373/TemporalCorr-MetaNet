"""Evaluation metrics for change detection.

Implements metrics from Section 6.3 of technical document:
- IoU / Jaccard Index
- F1 Score (Dice Coefficient equivalent)
- Precision / Recall
- Overall Accuracy
- kappa coefficient
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from collections import defaultdict


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """Compute confusion matrix elements.
    
    Args:
        pred: Predictions (B, 1, H, W) or (B, H, W)
        target: Ground truth (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binarization
    
    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    target = target.float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Compute confusion matrix
    tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
    
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute Intersection over Union (Jaccard Index).
    
    IoU = TP / (TP + FP + FN)
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
        smooth: Smoothing factor
    
    Returns:
        IoU score
    """
    cm = compute_confusion_matrix(pred, target, threshold)
    
    iou = cm["TP"] / (cm["TP"] + cm["FP"] + cm["FN"] + smooth)
    
    return iou


def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute mean IoU over both classes.
    
    mIoU = (IoU_change + IoU_nochange) / 2
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
        smooth: Smoothing factor
    
    Returns:
        mIoU score
    """
    cm = compute_confusion_matrix(pred, target, threshold)
    
    # IoU for change class
    iou_change = cm["TP"] / (cm["TP"] + cm["FP"] + cm["FN"] + smooth)
    
    # IoU for no-change class
    iou_nochange = cm["TN"] / (cm["TN"] + cm["FN"] + cm["FP"] + smooth)
    
    return (iou_change + iou_nochange) / 2


def compute_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute F1 Score (harmonic mean of precision and recall).
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
       = 2 * TP / (2*TP + FP + FN)
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
        smooth: Smoothing factor
    
    Returns:
        F1 score
    """
    cm = compute_confusion_matrix(pred, target, threshold)
    
    f1 = 2 * cm["TP"] / (2 * cm["TP"] + cm["FP"] + cm["FN"] + smooth)
    
    return f1


def compute_precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Tuple[float, float]:
    """Compute Precision and Recall.
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
        smooth: Smoothing factor
    
    Returns:
        Tuple of (precision, recall)
    """
    cm = compute_confusion_matrix(pred, target, threshold)
    
    precision = cm["TP"] / (cm["TP"] + cm["FP"] + smooth)
    recall = cm["TP"] / (cm["TP"] + cm["FN"] + smooth)
    
    return precision, recall


def compute_overall_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute Overall Accuracy.
    
    OA = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
    
    Returns:
        Overall accuracy
    """
    cm = compute_confusion_matrix(pred, target, threshold)
    
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    oa = (cm["TP"] + cm["TN"]) / total if total > 0 else 0.0
    
    return oa


def compute_kappa(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute Cohen's Kappa coefficient.
    
    κ = (p_o - p_e) / (1 - p_e)
    
    Where:
        p_o = observed agreement (accuracy)
        p_e = expected agreement by chance
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
        smooth: Smoothing factor
    
    Returns:
        Kappa coefficient
    """
    cm = compute_confusion_matrix(pred, target, threshold)
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    
    if total == 0:
        return 0.0
    
    # Observed agreement
    p_o = (cm["TP"] + cm["TN"]) / total
    
    # Expected agreement
    p_yes = ((cm["TP"] + cm["FP"]) / total) * ((cm["TP"] + cm["FN"]) / total)
    p_no = ((cm["TN"] + cm["FN"]) / total) * ((cm["TN"] + cm["FP"]) / total)
    p_e = p_yes + p_no
    
    # Kappa
    kappa = (p_o - p_e) / (1 - p_e + smooth)
    
    return kappa


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
    
    Returns:
        Dictionary with all metrics
    """
    cm = compute_confusion_matrix(pred, target, threshold)
    smooth = 1e-6
    
    precision = cm["TP"] / (cm["TP"] + cm["FP"] + smooth)
    recall = cm["TP"] / (cm["TP"] + cm["FN"] + smooth)
    
    return {
        "iou": compute_iou(pred, target, threshold),
        "miou": compute_miou(pred, target, threshold),
        "f1": compute_f1(pred, target, threshold),
        "precision": precision,
        "recall": recall,
        "accuracy": compute_overall_accuracy(pred, target, threshold),
        "kappa": compute_kappa(pred, target, threshold),
        "tp": cm["TP"],
        "tn": cm["TN"],
        "fp": cm["FP"],
        "fn": cm["FN"],
    }


class MetricTracker:
    """Tracks metrics over multiple batches/episodes.
    
    Accumulates statistics and computes aggregate metrics.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """Initialize tracker.
        
        Args:
            metrics: List of metric names to track
        """
        if metrics is None:
            metrics = ["iou", "miou", "f1", "precision", "recall", "accuracy", "kappa"]
        
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.values = defaultdict(list)
        self.confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ):
        """Update tracker with new predictions.
        
        Args:
            pred: Predictions
            target: Ground truth
            threshold: Binarization threshold
        """
        # Compute batch metrics
        batch_metrics = compute_all_metrics(pred, target, threshold)
        
        for name in self.metrics:
            if name in batch_metrics:
                self.values[name].append(batch_metrics[name])
        
        # Accumulate confusion matrix
        self.confusion["TP"] += batch_metrics["tp"]
        self.confusion["TN"] += batch_metrics["tn"]
        self.confusion["FP"] += batch_metrics["fp"]
        self.confusion["FN"] += batch_metrics["fn"]
    
    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics.
        
        Returns:
            Dictionary with average metrics and confusion-based metrics
        """
        results = {}
        
        # Average metrics
        for name, values in self.values.items():
            if values:
                results[f"{name}_avg"] = np.mean(values)
                results[f"{name}_std"] = np.std(values)
        
        # Confusion-based aggregate metrics
        smooth = 1e-6
        tp, tn, fp, fn = (
            self.confusion["TP"],
            self.confusion["TN"],
            self.confusion["FP"],
            self.confusion["FN"],
        )
        
        results["iou_global"] = tp / (tp + fp + fn + smooth)
        results["f1_global"] = 2 * tp / (2 * tp + fp + fn + smooth)
        results["precision_global"] = tp / (tp + fp + smooth)
        results["recall_global"] = tp / (tp + fn + smooth)
        
        total = tp + tn + fp + fn
        results["accuracy_global"] = (tp + tn) / total if total > 0 else 0
        
        return results
    
    def get_summary(self) -> str:
        """Get formatted summary string."""
        results = self.compute()
        
        lines = ["Metrics Summary:"]
        for k, v in sorted(results.items()):
            lines.append(f"  {k}: {v:.4f}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    pred = torch.rand(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    print("Single batch metrics:")
    metrics = compute_all_metrics(pred, target)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Test tracker
    print("\nTesting MetricTracker:")
    tracker = MetricTracker()
    
    for _ in range(5):
        pred = torch.rand(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        tracker.update(pred, target)
    
    print(tracker.get_summary())
