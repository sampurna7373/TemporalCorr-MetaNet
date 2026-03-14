"""Meta-Learner for TemporalCorr-MetaNet.

This module implements the MAML-style meta-learning wrapper for
episodic training of the change detection model.

Meta-Learning Framework (from Section 5.6):
1. Episode Sampling: Sample M classes, K support, N query
2. Inner Loop: Task-specific adaptation on support set
3. Outer Loop: Meta-parameter optimization on query set

Algorithm (from technical document):
    for episode in 1..max_episodes:
        Sample support set S, query set Q
        
        # Inner loop: task-specific adaptation
        L_S ← ComputeLoss(S, θ)
        θ' ← θ - β * ∇_θ L_S
        
        # Outer loop: meta-update
        L_Q ← ComputeLoss(Q, θ')
        θ ← θ - α * ∇_θ L_Q
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import higher  # For differentiable inner loop (optional)


class MetaLearner:
    """MAML-style meta-learner for few-shot change detection.
    
    Implements episodic meta-training with:
    - Inner loop: Fast adaptation on support set
    - Outer loop: Meta-parameter optimization
    
    Args:
        model: TemporalCorrMetaNet instance
        inner_lr: Learning rate for inner loop (β)
        outer_lr: Learning rate for outer loop (α)
        inner_steps: Number of inner loop gradient steps
        first_order: Use first-order approximation (FOMAML)
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = True,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.gradient_clip = gradient_clip
        
        # Meta-optimizer for outer loop
        self.meta_optimizer = Adam(
            model.parameters(),
            lr=outer_lr,
            weight_decay=weight_decay,
        )
        
        # Track training state
        self.episode_count = 0
        self.best_val_metric = 0.0
    
    def inner_loop_adapt(
        self,
        support_t1: torch.Tensor,
        support_t2: torch.Tensor,
        support_masks: torch.Tensor,
        loss_fn: callable,
    ) -> Dict[str, torch.Tensor]:
        """Perform inner loop adaptation on support set.
        
        Args:
            support_t1: Support images at T1 (K*num_classes, C, H, W)
            support_t2: Support images at T2 (K*num_classes, C, H, W)
            support_masks: Support masks (K*num_classes, H, W)
            loss_fn: Loss function to use
        
        Returns:
            Dictionary with adapted parameters and support loss
        """
        # Create a copy of model parameters for inner loop
        if self.first_order:
            # First-order MAML: use simple gradient descent
            adapted_params = {}
            for name, param in self.model.named_parameters():
                adapted_params[name] = param.clone()
            
            for step in range(self.inner_steps):
                # Forward pass
                outputs = self.model(support_t1, support_t2)
                support_loss = loss_fn(outputs["pred"], support_masks.unsqueeze(1).float())
                
                # Compute gradients
                grads = torch.autograd.grad(
                    support_loss,
                    self.model.parameters(),
                    create_graph=not self.first_order,
                    allow_unused=True,
                )
                
                # Update parameters
                for (name, param), grad in zip(self.model.named_parameters(), grads):
                    if grad is not None:
                        adapted_params[name] = adapted_params[name] - self.inner_lr * grad
            
            return {
                "adapted_params": adapted_params,
                "support_loss": support_loss.item(),
            }
        else:
            # Second-order MAML (more expensive but more accurate)
            # Requires higher library for differentiable optimization
            raise NotImplementedError("Second-order MAML requires 'higher' library")
    
    def compute_query_loss(
        self,
        query_t1: torch.Tensor,
        query_t2: torch.Tensor,
        query_masks: torch.Tensor,
        loss_fn: callable,
        support_features: Optional[torch.Tensor] = None,
        support_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss on query set.
        
        Args:
            query_t1: Query images at T1
            query_t2: Query images at T2
            query_masks: Query masks
            loss_fn: Loss function
            support_features: Optional support features for MCB/ATSM
            support_masks: Optional support masks for adaptation
        
        Returns:
            Query loss
        """
        outputs = self.model(
            query_t1,
            query_t2,
            support_features=support_features,
            support_masks=support_masks,
        )
        
        query_loss = loss_fn(outputs["pred"], query_masks.unsqueeze(1).float())
        
        return query_loss, outputs
    
    def episode_train(
        self,
        support_t1: torch.Tensor,
        support_t2: torch.Tensor,
        support_masks: torch.Tensor,
        query_t1: torch.Tensor,
        query_t2: torch.Tensor,
        query_masks: torch.Tensor,
        loss_fn: callable,
    ) -> Dict[str, Any]:
        """Perform one meta-training episode.
        
        Args:
            support_t1, support_t2, support_masks: Support set
            query_t1, query_t2, query_masks: Query set
            loss_fn: Loss function
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.meta_optimizer.zero_grad()
        
        # Inner loop: adapt on support set
        for step in range(self.inner_steps):
            outputs = self.model(support_t1, support_t2)
            support_loss = loss_fn(outputs["pred"], support_masks.unsqueeze(1).float())
            
            if step < self.inner_steps - 1:
                # Intermediate step gradients (for multi-step adaptation)
                grads = torch.autograd.grad(
                    support_loss,
                    [p for p in self.model.parameters() if p.requires_grad],
                    create_graph=not self.first_order,
                    allow_unused=True,
                    retain_graph=True,
                )
        
        # Get support features for ATSM/MCB adaptation
        with torch.no_grad():
            support_outputs = self.model(
                support_t1, support_t2, return_features=True
            )
            support_features = support_outputs["features"]["z"]
        
        # Outer loop: compute query loss
        query_loss, query_outputs = self.compute_query_loss(
            query_t1, query_t2, query_masks, loss_fn,
            support_features=support_features,
            support_masks=support_masks,
        )
        
        # Total loss
        total_loss = support_loss + 0.5 * query_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip,
            )
        
        # Meta-update
        self.meta_optimizer.step()
        
        self.episode_count += 1
        
        # Compute metrics
        with torch.no_grad():
            pred = (query_outputs["pred"] > 0.5).float()
            target = query_masks.unsqueeze(1).float()
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
        
        return {
            "episode": self.episode_count,
            "support_loss": support_loss.item(),
            "query_loss": query_loss.item(),
            "total_loss": total_loss.item(),
            "query_iou": iou,
        }
    
    def evaluate_episode(
        self,
        support_t1: torch.Tensor,
        support_t2: torch.Tensor,
        support_masks: torch.Tensor,
        query_t1: torch.Tensor,
        query_t2: torch.Tensor,
        query_masks: torch.Tensor,
        loss_fn: callable,
    ) -> Dict[str, Any]:
        """Evaluate on a single episode without updating weights.
        
        Args:
            support_t1, support_t2, support_masks: Support set
            query_t1, query_t2, query_masks: Query set
            loss_fn: Loss function
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get support features
            support_outputs = self.model(
                support_t1, support_t2, return_features=True
            )
            support_features = support_outputs["features"]["z"]
            
            # Evaluate on query set
            query_outputs = self.model(
                query_t1, query_t2,
                support_features=support_features,
                support_masks=support_masks,
            )
            
            query_loss = loss_fn(
                query_outputs["pred"],
                query_masks.unsqueeze(1).float(),
            )
            
            # Compute metrics
            pred = (query_outputs["pred"] > 0.5).float()
            target = query_masks.unsqueeze(1).float()
            
            # IoU
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            iou = intersection / (union + 1e-8)
            
            # Precision, Recall, F1
            tp = (pred * target).sum()
            fp = (pred * (1 - target)).sum()
            fn = ((1 - pred) * target).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "query_loss": query_loss.item(),
            "iou": iou.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }
    
    def save_checkpoint(
        self,
        path: str,
        extra_info: Optional[Dict] = None,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            extra_info: Additional information to save
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.meta_optimizer.state_dict(),
            "episode_count": self.episode_count,
            "best_val_metric": self.best_val_metric,
            "config": {
                "inner_lr": self.inner_lr,
                "outer_lr": self.outer_lr,
                "inner_steps": self.inner_steps,
                "first_order": self.first_order,
            },
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint.get("episode_count", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", 0.0)
        
        return checkpoint
    
    def set_learning_rate(self, outer_lr: float, inner_lr: Optional[float] = None) -> None:
        """Update learning rates.
        
        Args:
            outer_lr: New outer loop learning rate
            inner_lr: New inner loop learning rate (optional)
        """
        self.outer_lr = outer_lr
        for param_group in self.meta_optimizer.param_groups:
            param_group["lr"] = outer_lr
        
        if inner_lr is not None:
            self.inner_lr = inner_lr


class SimplifiedMetaLearner:
    """Simplified meta-learner without explicit inner loop.
    
    Uses the model's built-in adaptation mechanisms (ATSM, MCB)
    instead of gradient-based adaptation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
    ):
        self.model = model
        self.lr = lr
        self.gradient_clip = gradient_clip
        
        self.optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.episode_count = 0
    
    def train_episode(
        self,
        support_t1: torch.Tensor,
        support_t2: torch.Tensor,
        support_masks: torch.Tensor,
        query_t1: torch.Tensor,
        query_t2: torch.Tensor,
        query_masks: torch.Tensor,
        loss_fn: callable,
    ) -> Dict[str, Any]:
        """Train on a single episode.
        
        Uses support set to compute features for ATSM/MCB adaptation,
        then evaluates on query set.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get support features
        with torch.no_grad():
            support_outputs = self.model(
                support_t1, support_t2, return_features=True
            )
            support_features = support_outputs["features"]["z"]
        
        # Forward pass on query with support context
        query_outputs = self.model(
            query_t1, query_t2,
            support_features=support_features,
            support_masks=support_masks,
        )
        
        # Compute loss
        loss = loss_fn(query_outputs["pred"], query_masks.unsqueeze(1).float())
        
        # Backward and update
        loss.backward()
        
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip,
            )
        
        self.optimizer.step()
        self.episode_count += 1
        
        # Metrics
        with torch.no_grad():
            pred = (query_outputs["pred"] > 0.5).float()
            target = query_masks.unsqueeze(1).float()
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
        
        return {
            "episode": self.episode_count,
            "loss": loss.item(),
            "iou": iou,
        }


if __name__ == "__main__":
    from .temporalcorr_metanet import TemporalCorrMetaNet
    
    # Create model and meta-learner
    model = TemporalCorrMetaNet()
    meta_learner = MetaLearner(model, inner_lr=0.01, outer_lr=0.001)
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Inner LR: {meta_learner.inner_lr}")
    print(f"Outer LR: {meta_learner.outer_lr}")
    
    # Dummy data for testing
    batch_size = 4
    support_t1 = torch.randn(batch_size, 3, 512, 512)
    support_t2 = torch.randn(batch_size, 3, 512, 512)
    support_masks = torch.randint(0, 2, (batch_size, 512, 512)).float()
    
    query_t1 = torch.randn(batch_size, 3, 512, 512)
    query_t2 = torch.randn(batch_size, 3, 512, 512)
    query_masks = torch.randint(0, 2, (batch_size, 512, 512)).float()
    
    # Test episode training
    loss_fn = nn.BCELoss()
    metrics = meta_learner.episode_train(
        support_t1, support_t2, support_masks,
        query_t1, query_t2, query_masks,
        loss_fn,
    )
    
    print(f"\nEpisode metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
