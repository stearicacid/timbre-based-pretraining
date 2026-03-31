import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb
import logging
import os
from pathlib import Path
from typing import Tuple

from src.utils.training_metrics import TrainingMetrics
from src.utils.tradeoff_analysis import TradeoffAnalysis

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(
        self, 
        patience: int = 20, 
        min_delta: float = 1e-4,
        monitor: str = "val/loss",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, current_score: float) -> bool:
        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def setup_device(cfg: DictConfig) -> torch.device:
    """Setup device based on configuration."""
    if cfg.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.device)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    return device

def setup_wandb_for_sweep(cfg: DictConfig):
    """Sweep用のWandB初期化"""
    if cfg.logging.wandb.enabled:
        import wandb
        
        if wandb.run is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            base_name = getattr(cfg.logging.wandb, 'name', None) or cfg.experiment_name
            run_name = f"{base_name}_{timestamp}"
            
            wandb.init(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.get('entity', None),
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=cfg.logging.wandb.get('tags', []),
                notes=cfg.logging.wandb.get('notes', ''),
                mode=cfg.logging.wandb.get('mode', 'online')
            )
            logger.info(f"WandB run name: {run_name}")
        else:
            logger.info(f"WandB sweep run: {wandb.run.name}")
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

class EMAOptimizer:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name] + 
                                   (1 - self.decay) * param.data)
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    cfg: DictConfig,
    checkpoint_type: str = "best"
):
    """チェックポイントの保存（bestとfinalのみ）"""
    output_dir = Path(os.getcwd())
    
    if checkpoint_type == "best":
        filename = "best_model.pth"
    elif checkpoint_type == "final":
        filename = "final_model.pth"
    else:
        raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")
    
    filepath = output_dir / filename

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'model_beta': model.current_beta, 
        'model_use_kl': model.use_kl,     
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"{checkpoint_type.capitalize()} model saved: {filepath}")
    
    if (cfg.logging.wandb.enabled and 
        hasattr(cfg.logging, 'artifacts') and 
        cfg.logging.artifacts.get('log_model', False)):
        artifact = wandb.Artifact(
            name=f"{cfg.experiment_name}_{checkpoint_type}_model",
            type="model",
            description=f"{checkpoint_type.capitalize()} model checkpoint"
        )
        artifact.add_file(str(filepath))
        wandb.log_artifact(artifact)

def integrate_analysis_metrics(cfg: DictConfig) -> Tuple[TrainingMetrics, TradeoffAnalysis]:
    """分析メトリクスの初期化"""
    training_metrics = TrainingMetrics()
    tradeoff_analysis = TradeoffAnalysis()
    
    logger.info("Analysis metrics initialized")
    return training_metrics, tradeoff_analysis