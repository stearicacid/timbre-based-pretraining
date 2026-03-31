import hydra
from omegaconf import DictConfig
import torch
import logging
import os

from src.utils.reproducibility import set_seed, make_deterministic
from src.vae.trainer import Trainer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="vae")
def train(cfg: DictConfig) -> None:
    """Main training function."""

    logger.info(f"Working directory: {os.getcwd()}")

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

    trainer = Trainer.from_config(cfg, logger)
    trainer.fit()


if __name__ == "__main__":
    train()