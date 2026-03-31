import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb
import logging
import os
from pathlib import Path
import numpy as np

from src.vae.model import HarmonicVAE
from src.vae.dataset import create_dataloaders
from src.utils.reproducibility import set_seed, make_deterministic
from src.utils.logging_utils import log_model_info
from src.utils.beta_scheduler import BetaScheduler
from src.vae.io import save_checkpoint, EarlyStopping, setup_device, setup_wandb_for_sweep, EMAOptimizer, integrate_analysis_metrics
from src.vae.trainer import train_epoch, validate_epoch

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="vae")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    
    import os
    logger.info(f"Working directory: {os.getcwd()}")
        
    # === 1. 再現性の設定 ===
    logger.info("Setting up reproducibility...")
    set_seed(cfg.seed)
    if cfg.deterministic:
        try:
            make_deterministic()
            logger.info("Deterministic mode enabled")
        except Exception as e:
            logger.warning(f"Could not enable full deterministic mode: {e}")
            logger.info("Continuing with partial deterministic settings...")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # === 2. WandBの初期化（Sweep対応） ===
    setup_wandb_for_sweep(cfg)

    # === 3. デバイス設定 ===
    device = setup_device(cfg)
    
    # === 4. データローダー作成 ===
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        **cfg.dataset,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.get('prefetch_factor', 4),
    )

    sample_batch = next(iter(train_loader))
    
    if isinstance(sample_batch, (tuple, list)):
        if len(sample_batch) == 2:
            sample_data, sample_labels = sample_batch
            logger.info(f"Sample batch shape: {sample_data.shape}")
            logger.info(f"Sample labels shape: {sample_labels.shape}")
            sample_tensor = sample_data
        else:
            sample_tensor = sample_batch[0]
            logger.info(f"Sample batch shape: {sample_tensor.shape}")
    else:
        sample_tensor = sample_batch
        logger.info(f"Sample batch shape: {sample_tensor.shape}")
    
    # === 5. モデル初期化（修正版） ===
    logger.info("Initializing model...")

    model_params = {
        'input_dim': cfg.model.input_dim,
        'latent_dim': cfg.model.latent_dim,
        'hidden_dims': cfg.model.hidden_dims
        # 'contractive_weight': cfg.model.get('contractive_weight', 0.0),  # 追加
    }

    if hasattr(cfg.model, 'loss'):
        model_params['loss'] = OmegaConf.to_container(cfg.model.loss, resolve=True)
        logger.info(f"Model loss config: {model_params['loss']}")

    if hasattr(cfg.model, 'beta'):
        model_params['beta'] = cfg.model.beta
    
    model = HarmonicVAE(**model_params).to(device)
    
    model_mode = "AutoEncoder" if not model.use_kl else "VAE"
    logger.info(f"Model mode: {model_mode}")
    # logger.info(f"Model contractive weight: {model.contractive_weight}")
    if model.use_kl:
        logger.info(f"Initial β value: {model.current_beta}")
    else:
        logger.info("KL divergence disabled (AutoEncoder mode)")
    
    if cfg.logging.wandb.enabled:
        log_model_info(model, sample_tensor.to(device))

        wandb.config.update({
            "model_mode": model_mode, 
            "use_kl": model.use_kl,
            "actual_latent_dim": cfg.model.latent_dim,
            # "actual_contractive_weight": cfg.model.get('contractive_weight', 0.0),
            "actual_loss_weights": OmegaConf.to_container(cfg.model.loss.get('loss_weights', {}), resolve=True)
        })
    
    # === 6. 最適化設定 ===
    optimizer = hydra.utils.instantiate(
        cfg.optimization.optimizer, 
        params=model.parameters()
    )
    scheduler = hydra.utils.instantiate(
        cfg.optimization.scheduler, 
        optimizer=optimizer
    )
    
    logger.info(f"Optimizer weight decay: {optimizer.param_groups[0]['weight_decay']}")
    
    # === 7. Early stopping ===
    early_stopping = None
    if cfg.training.early_stopping.enabled:
        early_stopping_config = {
            k: v for k, v in cfg.training.early_stopping.items() 
            if k != 'enabled'
        }
        early_stopping = EarlyStopping(**early_stopping_config)

    # === βスケジューラーの初期化 ===
    beta_scheduler = None
    if (model.use_kl and 
        hasattr(cfg.model, 'beta_scheduler') and 
        cfg.model.beta_scheduler.enabled):
        
        freeze_epochs = cfg.model.beta_scheduler.get('freeze_epochs', 10)
        
        beta_scheduler = BetaScheduler(
            schedule_type=cfg.model.beta_scheduler.schedule_type,
            total_epochs=cfg.training.epochs,
            beta_start=cfg.model.beta_scheduler.beta_start,
            beta_end=cfg.model.beta_scheduler.beta_end,
            warmup_epochs=cfg.model.beta_scheduler.warmup_epochs,
            freeze_epochs=freeze_epochs  
        )
        
        logger.info(f"Beta scheduler initialized: {cfg.model.beta_scheduler.schedule_type}")
        logger.info(f"Freeze epochs (β=0): {freeze_epochs}")
        logger.info(f"Warmup epochs: {cfg.model.beta_scheduler.warmup_epochs}")
        
    elif not model.use_kl:
        logger.info("Beta scheduler disabled (AutoEncoder mode)")
    else:
        logger.info("Beta scheduler disabled by configuration")
    
    # === EMA optimizerの初期化 ===
    ema = None
    if cfg.optimization.get('use_ema', False):
        decay = cfg.optimization.ema.get('decay', 0.999)
        ema = EMAOptimizer(model, decay=decay)
        logger.info(f"EMA enabled with decay: {decay}")
        
        if cfg.logging.wandb.enabled:
            wandb.config.update({
                "ema_enabled": True,
                "ema_decay": decay
            })
    else:
        logger.info("EMA disabled")
        
        if cfg.logging.wandb.enabled:
            wandb.config.update({"ema_enabled": False})
    
    # === 分析メトリクスの初期化 ===
    training_metrics, tradeoff_analysis = integrate_analysis_metrics(cfg)
    
    # === 学習ループ ===
    logger.info("Starting training...")
    best_val_loss = float('inf')
    output_dir = Path(os.getcwd())
    
    for epoch in range(cfg.training.epochs):
        # βスケジューリング（VAEモードのみ）
        if beta_scheduler is not None and model.use_kl:
            current_beta = beta_scheduler.step(epoch)
            model.set_beta(current_beta)
            
            # より詳細なログ出力
            if epoch < beta_scheduler.freeze_epochs:
                logger.info(f"Epoch {epoch+1}: β = {current_beta:.4f} (FREEZE PHASE)")
            elif epoch < beta_scheduler.freeze_epochs + beta_scheduler.warmup_epochs:
                warmup_progress = (epoch - beta_scheduler.freeze_epochs) / (beta_scheduler.warmup_epochs - beta_scheduler.freeze_epochs)
                logger.info(f"Epoch {epoch+1}: β = {current_beta:.4f} (WARMUP: {warmup_progress:.1%})")
            else:
                logger.info(f"Epoch {epoch+1}: β = {current_beta:.4f} (STABLE)")
        
        # === 8.1 訓練フェーズ ===
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, cfg, epoch, ema
        )
        
        # === 8.2 検証フェーズ ===
        if (epoch + 1) % cfg.training.validation_interval == 0:
            val_metrics = validate_epoch(
                model, val_loader, device, cfg, epoch
            )
        else:
            val_metrics = {}
        
        # === 8.3 スケジューラーの更新 ===
        if val_metrics:
            scheduler.step(val_metrics['val/loss'])
        
        # === 8.4 ログ出力 ===
        log_metrics = {**train_metrics, **val_metrics}
        log_metrics['epoch'] = epoch + 1
        log_metrics['learning_rate'] = optimizer.param_groups[0]['lr']

        if model.use_kl:
            if beta_scheduler is not None:
                log_metrics['beta'] = current_beta
            else:
                log_metrics['beta'] = model.current_beta
        else:
            log_metrics['beta'] = 0.0 
        
        if cfg.logging.wandb.enabled:
            wandb.log(log_metrics)
        
        # === 8.5 ベストモデル保存 ===
        if val_metrics and val_metrics['val/loss'] < best_val_loss:
            best_val_loss = val_metrics['val/loss']
            save_checkpoint(model, optimizer, epoch, best_val_loss, cfg, checkpoint_type="best")
            logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
            
        # === 8.6 Early stopping ===
        if early_stopping and val_metrics:
            if early_stopping(val_metrics['val/loss']):
                logger.info(f"Early stopping at epoch {epoch+1} based on val/loss")
                break
        
    
    # === 9. 最終モデル保存 ===
    final_val_loss = val_metrics.get('val/loss', 0.0) if val_metrics else 0.0
    save_checkpoint(model, optimizer, epoch, final_val_loss, cfg, checkpoint_type="final")
    logger.info("Training completed. Final model saved.")
    
    # === 10. 最終処理 ===
    if cfg.logging.wandb.enabled:
        wandb.finish()
    
    logger.info("Training completed!")

    # === 学習終了後の分析 ===
    try:
        logger.info("Generating final analysis...")
        tradeoff_analysis.plot_tradeoff_curves(str(output_dir / "tradeoff_curves.png"))
        optimal_params = tradeoff_analysis.find_optimal_hyperparameters()
        
        logger.info("=== Final Analysis Results ===")
        if optimal_params:
            logger.info(f"Optimal epoch: {optimal_params.get('epoch', 'N/A')}")
            logger.info(f"Optimal reconstruction loss: {optimal_params.get('mean_recon_loss', 'N/A'):.4f}")
            logger.info(f"Optimal silhouette score: {optimal_params.get('silhouette_score', 'N/A'):.4f}")
            logger.info(f"Optimal ARI: {optimal_params.get('ari', 'N/A'):.4f}")
        else:
            logger.info("No optimal parameters found")
            
    except Exception as e:
        logger.warning(f"Error in final analysis: {e}")

# def log_reconstructions(
#     model: nn.Module,
#     val_loader: torch.utils.data.DataLoader,
#     device: torch.device,
#     epoch: int
# ):
#     """再構成結果をWandBにログ"""
#     model.eval()
#     with torch.no_grad():
#         sample_batch = next(iter(val_loader))

#         if isinstance(sample_batch, (tuple, list)):
#             if len(sample_batch) == 2:
#                 sample_data, _ = sample_batch  # (features, labels)
#             else:
#                 sample_data = sample_batch[0]
#         else:
#             sample_data = sample_batch

#         sample_data = sample_data[:8].to(device)  # [8, 45]
#         recon_sample, mu_sample, logvar_sample = model(sample_data)

#         from src.utils.normalization import ddsp_style_normalization
#         original_normalized = ddsp_style_normalization(sample_data[0])

#         original = original_normalized.cpu().numpy() 
#         reconstructed = recon_sample[0].cpu().numpy() 

#         mode_str = "AE" if not model.use_kl else "VAE"
#         if hasattr(model, 'use_triplet') and model.use_triplet:
#             mode_str += "+Triplet"
        
#         wandb.log({
#             'reconstructions': wandb.plot.line_series(
#                 xs=list(range(45)),
#                 ys=[original, reconstructed],
#                 keys=['Original', 'Reconstructed'],
#                 title=f"Harmonic Structure Reconstruction ({mode_str})",
#                 xname="Harmonic Number"
#             )
#         })

def log_latent_space(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int
):
    """潜在空間の分布をWandBにログ"""
    model.eval()
    all_mu = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 2:
                    data, _ = batch_data  # (features, labels)
                else:
                    data = batch_data[0]
            else:
                data = batch_data
                
            data = data.to(device)
            _, mu, _ = model(data)
            all_mu.append(mu.cpu().numpy())

    all_mu = np.concatenate(all_mu, axis=0)  # [total_samples, 10]
    
    mode_str = "AE" if not model.use_kl else "VAE"
    if hasattr(model, 'use_triplet') and model.use_triplet:
        mode_str += "+Triplet"
    
    for dim in range(all_mu.shape[1]):
        wandb.log({
            f'{mode_str.lower()}_latent_dim_{dim}': wandb.Histogram(all_mu[:, dim])
        })

if __name__ == "__main__":
    train()