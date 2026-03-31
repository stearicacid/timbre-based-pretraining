import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, Any
from pathlib import Path
import wandb
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

def setup_wandb(cfg: DictConfig) -> None:
    """
    WandBの初期化と設定
    
    Args:
        cfg: Hydra configuration
    """
    if not cfg.logging.wandb.enabled:
        logger.info("WandB logging is disabled")
        return
    
    # WandBの設定
    wandb_config = {
        "project": cfg.logging.wandb.project,
        "name": cfg.logging.wandb.name or cfg.experiment_name,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "tags": cfg.logging.wandb.tags,
        "notes": cfg.logging.wandb.notes,
        "mode": cfg.logging.wandb.mode,
    }
    
    # entityが設定されている場合のみ追加
    if cfg.logging.wandb.entity:
        wandb_config["entity"] = cfg.logging.wandb.entity
    
    # WandB初期化
    wandb.init(**wandb_config)
    
    # 設定ファイルをアーティファクトとして保存
    if cfg.logging.artifacts.log_config:
        log_config_artifact(cfg)
    
    logger.info(f"WandB initialized: {wandb.run.get_url()}")

def log_config_artifact(cfg: DictConfig) -> None:
    """
    設定ファイルをWandBアーティファクトとして保存
    
    Args:
        cfg: Hydra configuration
    """
    try:
        # 設定をYAMLファイルとして保存
        config_path = "config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)
        
        # アーティファクトとして登録
        artifact = wandb.Artifact(
            name=f"{cfg.experiment_name}_config",
            type="config",
            description="Experiment configuration"
        )
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)
        
        logger.info("Configuration saved as WandB artifact")
    except Exception as e:
        logger.warning(f"Failed to save config artifact: {e}")

def log_model_info(
    model: nn.Module, 
    sample_input: torch.Tensor,
    log_gradients: bool = True
) -> None:
    """
    モデルの情報をWandBにログ
    
    Args:
        model: PyTorchモデル
        sample_input: サンプル入力テンソル
        log_gradients: 勾配をログするかどうか
    """
    if not wandb.run:
        return
    
    try:
        # モデルアーキテクチャの詳細を取得
        model_info = get_model_summary(model, sample_input)
        
        # モデル情報をWandBにログ
        wandb.config.update({
            "model_info": model_info
        })
        
        # モデルの監視を開始（勾配とパラメータ）
        if log_gradients:
            wandb.watch(
                model, 
                log="all",  # log gradients and parameters
                log_freq=100,  # log every 100 batches
                log_graph=True  # log model graph
            )
        
        logger.info("Model information logged to WandB")
        
    except Exception as e:
        logger.warning(f"Failed to log model info: {e}")

def get_model_summary(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    モデルの詳細情報を取得
    
    Args:
        model: PyTorchモデル
        sample_input: サンプル入力テンソル
        
    Returns:
        モデル情報の辞書
    """
    # パラメータ数の計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # モデルサイズの計算（MB）
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    # 入力/出力サイズの取得
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input)
            if isinstance(output, tuple):
                output_shape = [out.shape for out in output]
            else:
                output_shape = output.shape
        except Exception as e:
            output_shape = f"Error: {str(e)}"
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "input_shape": list(sample_input.shape),
        "output_shape": output_shape,
        "architecture": str(model)
    }

def log_learning_curves(
    train_losses: Dict[str, list],
    val_losses: Dict[str, list],
    save_path: Optional[str] = None
) -> None:
    """
    学習曲線をプロットしてWandBにログ
    
    Args:
        train_losses: 訓練損失の履歴
        val_losses: 検証損失の履歴
        save_path: 保存パス（オプション）
    """
    if not wandb.run:
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Learning Curves', fontsize=16)
        
        # Total Loss
        axes[0, 0].plot(train_losses['total'], label='Train', alpha=0.7)
        axes[0, 0].plot(val_losses['total'], label='Validation', alpha=0.7)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction Loss
        axes[0, 1].plot(train_losses['recon'], label='Train', alpha=0.7)
        axes[0, 1].plot(val_losses['recon'], label='Validation', alpha=0.7)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL Divergence
        axes[1, 0].plot(train_losses['kl'], label='Train', alpha=0.7)
        axes[1, 0].plot(val_losses['kl'], label='Validation', alpha=0.7)
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss Ratio (KL/Reconstruction)
        train_ratio = np.array(train_losses['kl']) / (np.array(train_losses['recon']) + 1e-8)
        val_ratio = np.array(val_losses['kl']) / (np.array(val_losses['recon']) + 1e-8)
        axes[1, 1].plot(train_ratio, label='Train', alpha=0.7)
        axes[1, 1].plot(val_ratio, label='Validation', alpha=0.7)
        axes[1, 1].set_title('KL/Reconstruction Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # WandBにログ
        wandb.log({"learning_curves": wandb.Image(fig)})
        
        # ローカルに保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Failed to log learning curves: {e}")

def log_harmonic_analysis(
    original: np.ndarray,
    reconstructed: np.ndarray,
    epoch: int,
    sample_idx: int = 0
) -> None:
    """
    倍音構造の分析結果をWandBにログ
    
    Args:
        original: 元の倍音構造 [45]
        reconstructed: 再構成された倍音構造 [45]
        epoch: エポック番号
        sample_idx: サンプル番号
    """
    if not wandb.run:
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Harmonic Analysis - Epoch {epoch}, Sample {sample_idx}', fontsize=16)
        
        harmonic_numbers = np.arange(1, 46)
        
        # 元の倍音構造
        axes[0, 0].bar(harmonic_numbers, original, alpha=0.7, color='blue')
        axes[0, 0].set_title('Original Harmonic Structure')
        axes[0, 0].set_xlabel('Harmonic Number')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 再構成された倍音構造
        axes[0, 1].bar(harmonic_numbers, reconstructed, alpha=0.7, color='red')
        axes[0, 1].set_title('Reconstructed Harmonic Structure')
        axes[0, 1].set_xlabel('Harmonic Number')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 比較プロット
        axes[1, 0].plot(harmonic_numbers, original, 'o-', label='Original', alpha=0.7)
        axes[1, 0].plot(harmonic_numbers, reconstructed, 's-', label='Reconstructed', alpha=0.7)
        axes[1, 0].set_title('Comparison')
        axes[1, 0].set_xlabel('Harmonic Number')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 誤差分析
        error = np.abs(original - reconstructed)
        axes[1, 1].bar(harmonic_numbers, error, alpha=0.7, color='green')
        axes[1, 1].set_title(f'Absolute Error (MSE: {np.mean(error**2):.4f})')
        axes[1, 1].set_xlabel('Harmonic Number')
        axes[1, 1].set_ylabel('|Original - Reconstructed|')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # WandBにログ
        wandb.log({
            f"harmonic_analysis_epoch_{epoch}": wandb.Image(fig),
            f"reconstruction_mse_epoch_{epoch}": np.mean(error**2),
            f"reconstruction_mae_epoch_{epoch}": np.mean(error)
        })
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Failed to log harmonic analysis: {e}")

def log_latent_distribution(
    latent_samples: np.ndarray,
    epoch: int,
    labels: Optional[np.ndarray] = None
) -> None:
    """
    潜在空間の分布をWandBにログ
    
    Args:
        latent_samples: 潜在変数のサンプル [n_samples, latent_dim]
        epoch: エポック番号
        labels: ラベル（クラスタリング結果など）
    """
    if not wandb.run:
        return
    
    try:
        latent_dim = latent_samples.shape[1]
        
        # 2D散布図（最初の2次元）
        if latent_dim >= 2:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            if labels is not None:
                scatter = ax.scatter(latent_samples[:, 0], latent_samples[:, 1], 
                                   c=labels, alpha=0.6, cmap='tab10')
                plt.colorbar(scatter)
            else:
                ax.scatter(latent_samples[:, 0], latent_samples[:, 1], 
                          alpha=0.6, c='blue')
            
            ax.set_title(f'Latent Space Distribution - Epoch {epoch}')
            ax.set_xlabel('Latent Dimension 1')
            ax.set_ylabel('Latent Dimension 2')
            ax.grid(True, alpha=0.3)
            
            wandb.log({f"latent_space_2d_epoch_{epoch}": wandb.Image(fig)})
            plt.close()
        
        # 各次元のヒストグラム
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for dim in range(min(10, latent_dim)):
            axes[dim].hist(latent_samples[:, dim], bins=50, alpha=0.7, density=True)
            axes[dim].set_title(f'Dim {dim+1}')
            axes[dim].grid(True, alpha=0.3)
            
            # 正規分布との比較
            x = np.linspace(latent_samples[:, dim].min(), latent_samples[:, dim].max(), 100)
            mu, sigma = np.mean(latent_samples[:, dim]), np.std(latent_samples[:, dim])
            normal_dist = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            axes[dim].plot(x, normal_dist, 'r-', alpha=0.8, label='Normal')
            axes[dim].legend()
        
        # 未使用のサブプロットを非表示
        for dim in range(min(10, latent_dim), 10):
            axes[dim].set_visible(False)
        
        plt.suptitle(f'Latent Dimensions Distribution - Epoch {epoch}')
        plt.tight_layout()
        
        wandb.log({f"latent_histograms_epoch_{epoch}": wandb.Image(fig)})
        plt.close()
        
    except Exception as e:
        logger.warning(f"Failed to log latent distribution: {e}")

def log_reconstruction_metrics(
    original_batch: torch.Tensor,
    reconstructed_batch: torch.Tensor,
    prefix: str = "val"
) -> Dict[str, float]:
    """
    再構成メトリクスを計算してログ
    
    Args:
        original_batch: 元のバッチ [batch_size, 45]
        reconstructed_batch: 再構成されたバッチ [batch_size, 45]
        prefix: メトリクス名のプレフィックス
        
    Returns:
        計算されたメトリクス
    """
    with torch.no_grad():
        # NumPy配列に変換
        orig = original_batch.cpu().numpy()
        recon = reconstructed_batch.cpu().numpy()
        
        # メトリクス計算
        mse = np.mean((orig - recon) ** 2)
        mae = np.mean(np.abs(orig - recon))
        
        # 各倍音の相関
        correlations = []
        for i in range(orig.shape[1]):
            if np.std(orig[:, i]) > 1e-8 and np.std(recon[:, i]) > 1e-8:
                corr, _ = pearsonr(orig[:, i], recon[:, i])
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # R²スコア
        ss_res = np.sum((orig - recon) ** 2)
        ss_tot = np.sum((orig - np.mean(orig)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            f"{prefix}/reconstruction_mse": mse,
            f"{prefix}/reconstruction_mae": mae,
            f"{prefix}/reconstruction_correlation": avg_correlation,
            f"{prefix}/reconstruction_r2": r2_score
        }
        
        if wandb.run:
            wandb.log(metrics)
        
        return metrics

# ログ設定
def setup_logging(level: int = logging.INFO) -> None:
    """
    ログ設定のセットアップ
    
    Args:
        level: ログレベル
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )