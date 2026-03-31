from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb
import logging
import os
from pathlib import Path
import wandb
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_device(cfg: DictConfig) -> torch.device:
    """Setup device based on config"""
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

def setup_wandb(cfg: DictConfig):
    """Initialize WandB for sweep runs."""
    if cfg.logging.wandb.enabled:
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    cfg: DictConfig,
    checkpoint_type: str = "best"
):
    """Save checkpoints (best and final only)."""
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