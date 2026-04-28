import torch
import torch.nn as nn
import wandb
from torch.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime
import random
from tqdm.auto import tqdm

from model.architecture.gpt import GPT, GPTConfig 
from model.training.loss import cross_entropy_loss
from model.training.optimizer import create_optimizer, create_lr_scheduler
from model.training.dataset import TokenDataset

class Trainer:    
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
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.effective_batch_size = batch_size * accum_steps  
        self.grad_clip = grad_clip
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
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
        
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler(device=self.device) if use_mixed_precision else None
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.wandb_project = wandb_project
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _sample_batch(self, dataset: TokenDataset):
        dataset_size = len(dataset)
        if dataset_size <= 0:
            raise ValueError("Dataset is empty or block_size is too large for the data")

        if self.batch_size == 1:
            idx = random.randrange(dataset_size)
            x, y = dataset[idx]
            return x.unsqueeze(0), y.unsqueeze(0)

        indices = torch.randint(0, dataset_size, (self.batch_size,))
        xs = []
        ys = []
        for idx in indices.tolist():
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)

        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for start_idx in range(0, len(self.val_dataset), self.batch_size):
                batch_x = []
                batch_y = []

                for idx in range(start_idx, min(start_idx + self.batch_size, len(self.val_dataset))):
                    x, y = self.val_dataset[idx]
                    batch_x.append(x)
                    batch_y.append(y)

                x = torch.stack(batch_x, dim=0).to(self.device)
                y = torch.stack(batch_y, dim=0).to(self.device)

                with autocast(device_type=self.device, enabled=self.use_mixed_precision):
                    logits = self.model(x)
                    loss = cross_entropy_loss(logits, y)

                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, is_best: bool = False) -> str:
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")
    
    def train(self) -> None:
        wandb.init(project=self.wandb_project, name=f"mini-gpt-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.model.train()

        while self.global_step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True) 
            accum_loss = 0.0
            
            for _ in range(self.accum_steps):
                x, y = self._sample_batch(self.train_dataset)
                x, y = x.to(self.device), y.to(self.device)
                
                with autocast(device_type=self.device, enabled=self.use_mixed_precision):
                    logits = self.model(x)
                    loss = cross_entropy_loss(logits, y)
                    loss = loss / self.accum_steps
                
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accum_loss += loss.item()

            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.global_step += 1
            
            if self.global_step % 100 == 0:
                print(f"Step {self.global_step} | Loss: {accum_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                wandb.log({"train/loss": accum_loss, "train/grad_norm": grad_norm, "step": self.global_step})
            
            if self.global_step % 500 == 0:  
                val_loss = self.validate()
                print(f"Step {self.global_step} | Val Loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "step": self.global_step})
                
                self.save_checkpoint()
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

def train_gpt(model_config_dict: dict, train_data_path: str, val_data_path: str, resume_from=None, **trainer_kwargs):
    config = GPTConfig(**model_config_dict)
    model = GPT(config)
    
    train_dataset = TokenDataset(train_data_path, config.block_size)
    val_dataset = TokenDataset(val_data_path, config.block_size)
    
    trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, **trainer_kwargs)
    if resume_from:
        trainer.load_checkpoint(resume_from)
    trainer.train()