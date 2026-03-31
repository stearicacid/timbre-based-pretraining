import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb
import logging
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

from src.vae.io import save_checkpoint, EarlyStopping, setup_device, setup_wandb_for_sweep, EMAOptimizer, integrate_analysis_metrics

def train_epoch(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    epoch: int,
    ema: EMAOptimizer = None
) -> Dict[str, float]:
    """1エポックの訓練を実行"""
    model.train()
    train_losses = {'total': [], 'recon': [], 'kl': [], 'triplet': []}
    triplet_times = []  # 追加

    mode_str = "AE" if not model.use_kl else "VAE"
    if hasattr(model, 'use_triplet') and model.use_triplet:
        mode_str += "+Triplet"

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [{mode_str} Train]')

    for batch_idx, batch_data in enumerate(train_pbar):
        if isinstance(batch_data, (tuple, list)):
            if len(batch_data) == 2:
                data, labels = batch_data  
                data = data  
            else:
                data = batch_data[0]
                labels = None
        else:
            data = batch_data
            labels = None

        if not isinstance(data, torch.Tensor):
            logger.error(f"Expected tensor, got {type(data)}")
            continue

        data = data.to(device, non_blocking=True)
        if labels is not None:
            labels = labels.to(device, non_blocking=True)

        if data.size(0) == 1:
            continue
        
        optimizer.zero_grad()
        
        try:
            recon_data, mu, logvar = model(data)
            
            loss_weights = None
            if hasattr(cfg.model.loss, 'loss_weights'):
                loss_weights = OmegaConf.to_container(cfg.model.loss.loss_weights, resolve=True)

            losses = model.loss_function(
                recon_data, data, mu, logvar, 
                loss_weights=loss_weights, labels=labels, logger=logger
            )

            losses['total_loss'].backward()

            if (hasattr(cfg.optimization, 'gradient_clipping') and 
                cfg.optimization.gradient_clipping.enabled):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.optimization.gradient_clipping.get('max_norm', 1.0),
                    norm_type=cfg.optimization.gradient_clipping.get('norm_type', 2)
                )
            
            optimizer.step()
            
            if ema is not None:
                ema.update()

            train_losses['total'].append(losses['total_loss'].item())
            train_losses['recon'].append(losses['recon_loss'].item())
            train_losses['kl'].append(losses['kl_div'].item())
            train_losses['triplet'].append(losses.get('triplet_loss', torch.tensor(0.0)).item())
            triplet_times.append(losses.get('triplet_time', 0.0))  # 追加
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue

        if batch_idx % cfg.logging.log_interval == 0:
            postfix = {
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Recon': f"{losses['recon_loss'].item():.4f}",
                'Mode': losses.get('mode', mode_str)
            }
            
            if model.use_kl:
                postfix['KL'] = f"{losses['kl_div'].item():.4f}"
                postfix['β'] = f"{losses.get('beta', 0.0):.4f}"
            
            # Triplet lossの表示
            if hasattr(model, 'use_triplet') and model.use_triplet:
                postfix['Triplet'] = f"{losses.get('triplet_loss', torch.tensor(0.0)).item():.4f}"
            
            train_pbar.set_postfix(postfix)
        # prof.step()  # コメントアウト：プロファイラ無効化

    # triplet_timeの平均をログ出力
    if triplet_times:
        logger.info(f"[Triplet profiling] Average triplet_time per batch: {np.mean(triplet_times):.4f} sec")

    return {
        'train/loss': np.mean(train_losses['total']) if train_losses['total'] else 0.0,
        'train/recon_loss': np.mean(train_losses['recon']) if train_losses['recon'] else 0.0,
        'train/kl_div': np.mean(train_losses['kl']) if train_losses['kl'] else 0.0,
        'train/triplet_loss': np.mean(train_losses['triplet']) if train_losses['triplet'] else 0.0,
        'train/triplet_time': np.mean(triplet_times) if triplet_times else 0.0  # 追加
    }

def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: DictConfig,
    epoch: int
) -> Dict[str, float]:
    """1エポックの検証を実行"""
    model.eval()
    val_losses = {'total': [], 'recon': [], 'kl': [], 'triplet': []}
    
    mode_str = "AE" if not model.use_kl else "VAE"
    if hasattr(model, 'use_triplet') and model.use_triplet:
        mode_str += "+Triplet"
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [{mode_str} Val]')
        
        for batch_idx, batch_data in enumerate(val_pbar):
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 2:
                    data, labels = batch_data
                    data = data
                else:
                    data = batch_data[0]
                    labels = None
            else:
                data = batch_data
                labels = None
            
            data = data.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)

            if data.size(0) == 1:
                continue
            
            try:
                recon_data, mu, logvar = model(data)

                loss_weights = None
                if hasattr(cfg.model.loss, 'loss_weights'):
                    loss_weights = OmegaConf.to_container(cfg.model.loss.loss_weights, resolve=True)
                
                losses = model.loss_function(recon_data, data, mu, logvar, loss_weights=loss_weights, labels=labels)
                
                val_losses['total'].append(losses['total_loss'].item())
                val_losses['recon'].append(losses['recon_loss'].item())
                val_losses['kl'].append(losses['kl_div'].item())
                val_losses['triplet'].append(losses.get('triplet_loss', torch.tensor(0.0)).item())
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue

        # try:
        #     if (cfg.logging.wandb.enabled and 
        #         hasattr(cfg.logging, 'log_reconstruction_freq') and
        #         epoch % cfg.logging.log_reconstruction_freq == 0):
        #         log_reconstructions(model, val_loader, device, epoch)
            
        #     if (cfg.logging.wandb.enabled and 
        #         hasattr(cfg.logging, 'log_latent_freq') and
        #         epoch % cfg.logging.log_latent_freq == 0):
        #         log_latent_space(model, val_loader, device, epoch)
        # except Exception as e:
        #     logger.warning(f"Error in visualization logging: {e}")
    
    return {
        'val/loss': np.mean(val_losses['total']) if val_losses['total'] else 0.0,
        'val/recon_loss': np.mean(val_losses['recon']) if val_losses['recon'] else 0.0,
        'val/kl_div': np.mean(val_losses['kl']) if val_losses['kl'] else 0.0,
        'val/triplet_loss': np.mean(val_losses['triplet']) if val_losses['triplet'] else 0.0
    }