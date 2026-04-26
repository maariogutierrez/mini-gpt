import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import math
from pathlib import Path
from datetime import datetime

from model.architecture.gpt import GPT
from model.training.loss import cross_entropy_loss
from model.training.optimizer import create_optimizer, create_lr_scheduler
from model.training.dataset import TokenDataset


class Trainer:
    """
    Advanced training loop with gradient accumulation, mixed precision, 
    gradient clipping, validation, and W&B logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: TokenDataset,
        val_dataset: TokenDataset,
        batch_size: int = 16,
        accum_steps: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        grad_clip: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        wandb_project: str = "mini-gpt",
        use_mixed_precision: bool = True,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: GPT model instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Micro-batch size (16)
            accum_steps: Gradient accumulation steps (32 for 512 effective batch)
            learning_rate: Base learning rate
            weight_decay: AdamW weight decay
            warmup_steps: Learning rate warmup steps
            max_steps: Total training steps
            grad_clip: Gradient clipping threshold (1.0)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            wandb_project: W&B project name
            use_mixed_precision: Enable mixed precision training (AMP)
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        # Training hyperparameters
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.effective_batch_size = batch_size * accum_steps  # 512
        self.grad_clip = grad_clip
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps
        )
        
        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Logging
        self.wandb_project = wandb_project
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
    
    def compute_grad_norm(self) -> float:
        """Compute and return the norm of all gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def validate(self) -> float:
        """
        Run validation on the validation dataset.
        
        Returns:
            Mean validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model(x)
                        loss = cross_entropy_loss(logits, y)
                else:
                    logits = self.model(x)
                    loss = cross_entropy_loss(logits, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return val_loss
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            is_best: If True, also save as 'best_model.pt'
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")
    
    def train(self) -> None:
        """
        Main training loop with gradient accumulation, mixed precision,
        validation, and W&B logging.
        """
        # Initialize W&B
        wandb.init(
            project=self.wandb_project,
            name=f"mini-gpt-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "batch_size": self.batch_size,
                "accum_steps": self.accum_steps,
                "effective_batch_size": self.effective_batch_size,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "grad_clip": self.grad_clip,
                "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps,
                "mixed_precision": self.use_mixed_precision,
            },
        )
        
        self.model.train()
        accum_loss = 0.0
        num_batches_in_accum = 0
        
        print(f"Starting training: {self.effective_batch_size} effective batch size")
        print(f"({self.batch_size} micro-batch × {self.accum_steps} accum steps)")
        print(f"Device: {self.device}\n")
        
        # Create iterator that repeats as needed
        train_iter = iter(self.train_loader)
        
        while self.global_step < self.max_steps:
            # Gradient accumulation loop
            for accum_step in range(self.accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    # Reset iterator when epoch ends
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model(x)
                        loss = cross_entropy_loss(logits, y)
                        # Scale loss for gradient accumulation
                        loss = loss / self.accum_steps
                else:
                    logits = self.model(x)
                    loss = cross_entropy_loss(logits, y)
                    loss = loss / self.accum_steps
                
                # Backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accum_loss += loss.item()
                num_batches_in_accum += 1
            
            # Gradient clipping (before optimizer step)
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            grad_norm = self.compute_grad_norm()
            
            # Optimizer step
            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1
            
            # Compute average loss over accumulation steps
            avg_train_loss = accum_loss / num_batches_in_accum
            accum_loss = 0.0
            num_batches_in_accum = 0
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log every step to W&B
            wandb.log({
                "train/loss": avg_train_loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": current_lr,
                "step": self.global_step,
            }, step=self.global_step)
            
            # Validation every 500 steps
            if self.global_step % 500 == 0:
                val_loss = self.validate()
                wandb.log({
                    "val/loss": val_loss,
                    "step": self.global_step,
                }, step=self.global_step)
                
                # Save checkpoint if validation loss improved
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Save checkpoint
                self.save_checkpoint(is_best=is_best)
                
                print(
                    f"Step {self.global_step:6d} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Grad Norm: {grad_norm:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Print progress every 100 steps
            elif self.global_step % 100 == 0:
                print(
                    f"Step {self.global_step:6d} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Grad Norm: {grad_norm:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
        
        print(f"\nTraining complete! Final step: {self.global_step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        wandb.finish()


def train_gpt(
    model_config: dict,
    train_data_path: str,
    val_data_path: str,
    output_dir: str = "./outputs",
    **trainer_kwargs
) -> Trainer:
    """
    Convenience function to create and train a GPT model.
    
    Args:
        model_config: Configuration dict for GPT model
        train_data_path: Path to training .bin file
        val_data_path: Path to validation .bin file
        output_dir: Directory for checkpoints and outputs
        **trainer_kwargs: Additional arguments for Trainer
        
    Returns:
        Trained Trainer instance
    """
    # Create model
    model = GPT(**model_config)
    device = trainer_kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    block_size = model_config.get('block_size', 1024)
    train_dataset = TokenDataset(train_data_path, block_size)
    val_dataset = TokenDataset(val_data_path, block_size)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=output_dir,
        **trainer_kwargs
    )
    
    # Train
    trainer.train()
    
    return trainer
