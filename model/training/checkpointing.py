"""Checkpoint management for model training with auto-save support."""

import os
import torch
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import asdict


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup to save storage space.
    
    Keeps only the last N checkpoints to prevent filling up Google Drive quota.
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints (e.g., '/content/drive/MyDrive/checkpoints')
            max_checkpoints: Maximum number of checkpoints to keep (default: 3 for Colab)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
    
    def _get_checkpoint_number(self, filename: str) -> Optional[int]:
        """Extract checkpoint number from filename.
        
        Expected format: checkpoint_step_000500.pt
        """
        try:
            if filename.startswith('checkpoint_step_') and filename.endswith('.pt'):
                step = int(filename.replace('checkpoint_step_', '').replace('.pt', ''))
                return step
            return None
        except (ValueError, AttributeError):
            return None
    
    def _cleanup_old_checkpoints(self):
        """Delete oldest checkpoints, keeping only max_checkpoints most recent."""
        checkpoints = []
        
        # Collect all checkpoint files with their steps
        for file in self.checkpoint_dir.glob('checkpoint_step_*.pt'):
            step = self._get_checkpoint_number(file.name)
            if step is not None:
                checkpoints.append((step, file))
        
        # Sort by step and delete oldest if over limit
        if len(checkpoints) > self.max_checkpoints:
            checkpoints.sort(key=lambda x: x[0])  # Sort by step
            for step, file_path in checkpoints[:-self.max_checkpoints]:
                file_path.unlink()
                print(f"Deleted old checkpoint: {file_path.name}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        config: Any,
    ) -> str:
        """Save model checkpoint with all necessary state.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer with state to save
            step: Current training step
            loss: Current loss value
            config: Model config object (dataclass with __dict__ or asdict support)
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step:06d}.pt'
        
        # Prepare config dict
        if hasattr(config, '__dict__'):
            config_dict = vars(config)
        else:
            try:
                config_dict = asdict(config)
            except TypeError:
                config_dict = config if isinstance(config, dict) else {}
        
        # Create checkpoint dictionary
        checkpoint = {
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config_dict,
        }
        
        # Save to disk
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path} (step {step}, loss {loss:.4f})")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu',
    ) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float, Dict[str, Any]]:
        """Load checkpoint and restore model, optimizer, and training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            device: Device to load onto ('cpu', 'cuda', etc.)
        
        Returns:
            Tuple of (model, optimizer, step, loss, config)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Restore optimizer state if provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        step = checkpoint['step']
        loss = checkpoint['loss']
        config = checkpoint['config']
        
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  Step: {step}, Loss: {loss:.4f}")
        
        return model, optimizer, step, loss, config
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint in the directory.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = []
        
        for file in self.checkpoint_dir.glob('checkpoint_step_*.pt'):
            step = self._get_checkpoint_number(file.name)
            if step is not None:
                checkpoints.append((step, str(file)))
        
        if not checkpoints:
            return None
        
        # Return path of checkpoint with highest step
        return max(checkpoints, key=lambda x: x[0])[1]


class AutoSaver:
    """Context manager for automatic checkpoint saving during training.
    
    Saves every N steps without cluttering your code.
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,
        save_every_n_steps: int = 500,
    ):
        """Initialize auto-saver.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            model: Model being trained
            optimizer: Optimizer being used
            config: Model configuration
            save_every_n_steps: Save checkpoint every N steps (default: 500 for Colab)
        """
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.save_every_n_steps = save_every_n_steps
    
    def maybe_save(self, step: int, loss: float):
        """Save checkpoint if it's time.
        
        Args:
            step: Current training step
            loss: Current loss value
        
        Returns:
            True if checkpoint was saved, False otherwise
        """
        if step > 0 and step % self.save_every_n_steps == 0:
            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                step,
                loss,
                self.config,
            )
            return True
        return False


# Convenience functions for simple use cases

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str,
    config: Optional[Any] = None,
) -> str:
    """Simple function to save a checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer with state
        step: Current training step
        loss: Current loss value
        path: Path or directory to save to
        config: Optional model config
    
    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(str(Path(path).parent))
    return manager.save_checkpoint(model, optimizer, step, loss, config)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float, Dict[str, Any]]:
    """Simple function to load a checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load into
        optimizer: Optional optimizer to load into
        device: Device to load onto
    
    Returns:
        Tuple of (model, optimizer, step, loss, config)
    """
    manager = CheckpointManager(str(Path(path).parent))
    return manager.load_checkpoint(path, model, optimizer, device)
