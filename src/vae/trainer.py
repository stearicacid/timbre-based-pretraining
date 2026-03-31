import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils.beta_scheduler import BetaScheduler
from src.utils.logging import log_model_info
from src.vae.dataset import create_dataloaders
from src.vae.hooks import EMAOptimizer, EarlyStopping
from src.vae.io import (
    integrate_analysis_metrics,
    save_checkpoint,
    setup_device,
    setup_wandb_for_sweep,
)
from src.vae.model import HarmonicVAE


class Trainer:
    """Stateful trainer that owns full VAE training lifecycle."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger,
        device: torch.device,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        early_stopping: Optional[EarlyStopping],
        beta_scheduler: Optional[BetaScheduler],
        ema: Optional[EMAOptimizer],
        output_dir: Path,
        training_metrics,
        tradeoff_analysis,
        loss_weights: Optional[dict],
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.beta_scheduler = beta_scheduler
        self.ema = ema
        self.best_val_loss = float("inf")
        self.output_dir = output_dir
        self.training_metrics = training_metrics
        self.tradeoff_analysis = tradeoff_analysis
        self.loss_weights = loss_weights

    @classmethod
    def from_config(cls, cfg: DictConfig, logger: Optional[logging.Logger] = None) -> "Trainer":
        logger = logger or logging.getLogger(__name__)

        setup_wandb_for_sweep(cfg)
        device = setup_device(cfg)

        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            **cfg.dataset,
            num_workers=cfg.training.dataloader_num_workers,
            pin_memory=cfg.training.pin_memory,
            prefetch_factor=cfg.training.get("prefetch_factor", 4),
        )

        sample_tensor = cls._extract_sample_tensor(next(iter(train_loader)), logger)

        logger.info("Initializing model...")
        model_params = {
            "input_dim": cfg.model.input_dim,
            "latent_dim": cfg.model.latent_dim,
            "hidden_dims": cfg.model.hidden_dims,
        }
        if hasattr(cfg.model, "loss"):
            model_params["loss"] = OmegaConf.to_container(cfg.model.loss, resolve=True)
            logger.info(f"Model loss config: {model_params['loss']}")
        if hasattr(cfg.model, "beta"):
            model_params["beta"] = cfg.model.beta

        model = HarmonicVAE(**model_params).to(device)

        model_mode = "VAE+Triplet"
        logger.info(f"Model mode: {model_mode}")
        logger.info(f"Initial β value: {model.current_beta}")

        if cfg.logging.wandb.enabled:
            log_model_info(model, sample_tensor.to(device))
            wandb.config.update(
                {
                    "model_mode": model_mode,
                    "actual_latent_dim": cfg.model.latent_dim,
                    "actual_loss_weights": OmegaConf.to_container(
                        cfg.model.loss.get("loss_weights", {}), resolve=True
                    ),
                }
            )

        optimizer = hydra.utils.instantiate(cfg.optimization.optimizer, params=model.parameters())
        scheduler = hydra.utils.instantiate(cfg.optimization.scheduler, optimizer=optimizer)
        logger.info(f"Optimizer weight decay: {optimizer.param_groups[0]['weight_decay']}")

        early_stopping = None
        if cfg.training.early_stopping.enabled:
            early_stopping_config = {
                k: v for k, v in cfg.training.early_stopping.items() if k != "enabled"
            }
            early_stopping = EarlyStopping(**early_stopping_config)

        beta_scheduler = None
        if hasattr(cfg.model, "beta_scheduler") and cfg.model.beta_scheduler.enabled:
            freeze_epochs = cfg.model.beta_scheduler.get("freeze_epochs", 10)
            beta_scheduler = BetaScheduler(
                schedule_type=cfg.model.beta_scheduler.schedule_type,
                total_epochs=cfg.training.epochs,
                beta_start=cfg.model.beta_scheduler.beta_start,
                beta_end=cfg.model.beta_scheduler.beta_end,
                warmup_epochs=cfg.model.beta_scheduler.warmup_epochs,
                freeze_epochs=freeze_epochs,
            )
            logger.info(f"Beta scheduler initialized: {cfg.model.beta_scheduler.schedule_type}")
            logger.info(f"Freeze epochs (β=0): {freeze_epochs}")
            logger.info(f"Warmup epochs: {cfg.model.beta_scheduler.warmup_epochs}")
        else:
            logger.info("Beta scheduler disabled by configuration")

        ema = None
        if cfg.optimization.get("use_ema", False):
            decay = cfg.optimization.ema.get("decay", 0.999)
            ema = EMAOptimizer(model, decay=decay)
            logger.info(f"EMA enabled with decay: {decay}")
            if cfg.logging.wandb.enabled:
                wandb.config.update({"ema_enabled": True, "ema_decay": decay})
        else:
            logger.info("EMA disabled")
            if cfg.logging.wandb.enabled:
                wandb.config.update({"ema_enabled": False})

        training_metrics, tradeoff_analysis = integrate_analysis_metrics(cfg)
        loss_weights = cls._resolve_loss_weights(cfg)
        output_dir = Path(os.getcwd())

        logger.info(f"Trainer initialized on {device}")
        return cls(
            cfg=cfg,
            logger=logger,
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            beta_scheduler=beta_scheduler,
            ema=ema,
            output_dir=output_dir,
            training_metrics=training_metrics,
            tradeoff_analysis=tradeoff_analysis,
            loss_weights=loss_weights,
        )

    def fit(self) -> None:
        self.logger.info("Starting training...")
        val_metrics: Dict[str, float] = {}

        for epoch in range(self.cfg.training.epochs):
            current_beta = self._apply_beta_schedule(epoch)

            train_metrics = self.train_epoch(epoch)
            if (epoch + 1) % self.cfg.training.validation_interval == 0:
                val_metrics = self.validate_epoch(epoch)
            else:
                val_metrics = {}

            if val_metrics:
                self.scheduler.step(val_metrics["val/loss"])

            log_metrics = {**train_metrics, **val_metrics}
            log_metrics["epoch"] = epoch + 1
            log_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            log_metrics["beta"] = current_beta

            if self.cfg.logging.wandb.enabled:
                wandb.log(log_metrics)

            if val_metrics and val_metrics["val/loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val/loss"]
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_val_loss,
                    self.cfg,
                    checkpoint_type="best",
                )
                self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")

            if self.early_stopping and val_metrics and self.early_stopping(val_metrics["val/loss"]):
                self.logger.info(f"Early stopping at epoch {epoch+1} based on val/loss")
                break

        final_val_loss = val_metrics.get("val/loss", 0.0) if val_metrics else 0.0
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            final_val_loss,
            self.cfg,
            checkpoint_type="final",
        )
        self.logger.info("Training completed. Final model saved.")

        if self.cfg.logging.wandb.enabled:
            wandb.finish()

        self.logger.info("Training completed!")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        train_losses = {"total": [], "recon": [], "kl": [], "triplet": []}
        triplet_times = []
        mode_str = self._get_mode_str()

        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [{mode_str} Train]")

        for batch_idx, batch_data in enumerate(train_pbar):
            data, labels = self._prepare_batch(batch_data)
            # if data is None:
            #     continue

            self.optimizer.zero_grad()
            recon_data, mu, logvar = self.model(data)
            losses = self.model.loss_function(
                recon_data,
                data,
                mu,
                logvar,
                loss_weights=self.loss_weights,
                labels=labels,
                logger=self.logger,
            )
            losses["total_loss"].backward()

            if (
                hasattr(self.cfg.optimization, "gradient_clipping")
                and self.cfg.optimization.gradient_clipping.enabled
            ):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.optimization.gradient_clipping.get("max_norm", 1.0),
                    norm_type=self.cfg.optimization.gradient_clipping.get("norm_type", 2),
                )

            self.optimizer.step()
            if self.ema is not None:
                self.ema.update()

            train_losses["total"].append(losses["total_loss"].item())
            train_losses["recon"].append(losses["recon_loss"].item())
            train_losses["kl"].append(losses["kl_div"].item())
            train_losses["triplet"].append(
                losses.get("triplet_loss", torch.tensor(0.0)).item()
            )
            triplet_times.append(losses.get("triplet_time", 0.0))


            if batch_idx % self.cfg.logging.log_interval == 0:
                postfix = {
                    "Loss": f"{losses['total_loss'].item():.4f}",
                    "Recon": f"{losses['recon_loss'].item():.4f}",
                    "Mode": losses.get("mode", mode_str),
                    "KL": f"{losses['kl_div'].item():.4f}",
                    "β": f"{losses.get('beta', 0.0):.4f}",
                    "Triplet": f"{losses.get('triplet_loss', torch.tensor(0.0)).item():.4f}",
                }
                train_pbar.set_postfix(postfix)

        if triplet_times:
            self.logger.info(
                "[Triplet profiling] Average triplet_time per batch: "
                f"{np.mean(triplet_times):.4f} sec"
            )

        return {
            "train/loss": np.mean(train_losses["total"]) if train_losses["total"] else 0.0,
            "train/recon_loss": np.mean(train_losses["recon"]) if train_losses["recon"] else 0.0,
            "train/kl_div": np.mean(train_losses["kl"]) if train_losses["kl"] else 0.0,
            "train/triplet_loss": np.mean(train_losses["triplet"])
            if train_losses["triplet"]
            else 0.0,
            "train/triplet_time": np.mean(triplet_times) if triplet_times else 0.0,
        }

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        val_losses = {"total": [], "recon": [], "kl": [], "triplet": []}
        mode_str = self._get_mode_str()

        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [{mode_str} Val]")

            for batch_idx, batch_data in enumerate(val_pbar):
                data, labels = self._prepare_batch(batch_data)
                if data is None:
                    continue

                recon_data, mu, logvar = self.model(data)
                losses = self.model.loss_function(
                    recon_data,
                    data,
                    mu,
                    logvar,
                    loss_weights=self.loss_weights,
                    labels=labels,
                )

                val_losses["total"].append(losses["total_loss"].item())
                val_losses["recon"].append(losses["recon_loss"].item())
                val_losses["kl"].append(losses["kl_div"].item())
                val_losses["triplet"].append(
                    losses.get("triplet_loss", torch.tensor(0.0)).item()
                )

        return {
            "val/loss": np.mean(val_losses["total"]) if val_losses["total"] else 0.0,
            "val/recon_loss": np.mean(val_losses["recon"]) if val_losses["recon"] else 0.0,
            "val/kl_div": np.mean(val_losses["kl"]) if val_losses["kl"] else 0.0,
            "val/triplet_loss": np.mean(val_losses["triplet"]) if val_losses["triplet"] else 0.0,
        }

    @staticmethod
    def _resolve_loss_weights(cfg: DictConfig) -> Optional[dict]:
        if hasattr(cfg.model, "loss") and hasattr(cfg.model.loss, "loss_weights"):
            return OmegaConf.to_container(cfg.model.loss.loss_weights, resolve=True)
        return None

    @staticmethod
    def _extract_sample_tensor(sample_batch, logger: logging.Logger) -> torch.Tensor:
        if isinstance(sample_batch, (tuple, list)):
            if len(sample_batch) == 2:
                sample_data, sample_labels = sample_batch
                logger.info(f"Sample batch shape: {sample_data.shape}")
                logger.info(f"Sample labels shape: {sample_labels.shape}")
                return sample_data
            sample_tensor = sample_batch[0]
            logger.info(f"Sample batch shape: {sample_tensor.shape}")
            return sample_tensor

        logger.info(f"Sample batch shape: {sample_batch.shape}")
        return sample_batch

    def _unpack_batch(self, batch_data) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(batch_data, (tuple, list)):
            if len(batch_data) == 2:
                return batch_data[0], batch_data[1]
            return batch_data[0], None
        return batch_data, None

    def _move_batch_to_device(
        self, data: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        data = data.to(self.device, non_blocking=True)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)
        return data, labels

    def _prepare_batch(self, batch_data) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        data, labels = self._unpack_batch(batch_data)
        if not isinstance(data, torch.Tensor):
            self.logger.error(f"Expected tensor, got {type(data)}")
            return None, None

        data, labels = self._move_batch_to_device(data, labels)
        if data.size(0) == 1:
            return None, None
        return data, labels

    def _get_mode_str(self) -> str:
        return "VAE+Triplet"

    def _apply_beta_schedule(self, epoch: int) -> float:
        if self.beta_scheduler is not None:
            current_beta = self.beta_scheduler.step(epoch)
            self.model.set_beta(current_beta)
            if epoch < self.beta_scheduler.freeze_epochs:
                self.logger.info(
                    f"Epoch {epoch+1}: β = {current_beta:.4f} (FREEZE PHASE)"
                )
            elif epoch < self.beta_scheduler.freeze_epochs + self.beta_scheduler.warmup_epochs:
                warmup_progress = (
                    (epoch - self.beta_scheduler.freeze_epochs)
                    / (self.beta_scheduler.warmup_epochs - self.beta_scheduler.freeze_epochs)
                )
                self.logger.info(
                    f"Epoch {epoch+1}: β = {current_beta:.4f} "
                    f"(WARMUP: {warmup_progress:.1%})"
                )
            else:
                self.logger.info(f"Epoch {epoch+1}: β = {current_beta:.4f} (STABLE)")
            return current_beta
        return self.model.current_beta