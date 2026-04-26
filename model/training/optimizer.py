import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(model, learning_rate=1e-4, weight_decay=0.01):
    """
    Create AdamW optimizer with separate parameter groups.
    
    - Biases and LayerNorm parameters: no weight decay
    - All other parameters: weight decay applied
    
    Args:
        model: PyTorch model
        learning_rate: Base learning rate
        weight_decay: Weight decay coefficient (applied to non-bias/LayerNorm params)
    
    Returns:
        AdamW optimizer with configured parameter groups
    """
    # Separate parameters into two groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for biases and LayerNorm parameters
        if "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(param_groups, lr=learning_rate)
    return optimizer


def create_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr=0.0):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Schedule:
    - Steps 0 → warmup_steps: LR increases linearly from 0 to base_lr
    - Steps warmup_steps → max_steps: LR follows cosine decay to min_lr
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for linear warmup
        max_steps: Total number of training steps
        min_lr: Minimum learning rate (as fraction of base_lr)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


class WarmupCosineScheduler:
    """
    Wrapper class for learning rate scheduling with warmup and cosine decay.
    Provides additional utilities for tracking learning rate.
    """
    
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0.0):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of steps for linear warmup
            max_steps: Total number of training steps
            min_lr: Minimum learning rate (as fraction of base_lr)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # Get base learning rates from optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = float(self.current_step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            lr_scale = max(self.min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Update learning rates for all parameter groups
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * lr_scale
    
    def get_last_lr(self):
        """Return current learning rate."""
        return [group["lr"] for group in self.optimizer.param_groups]
    
    def get_progress(self):
        """Return current training progress as fraction of max_steps."""
        return min(1.0, self.current_step / max(1, self.max_steps))
