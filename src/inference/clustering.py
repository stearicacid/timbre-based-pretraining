import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.vae.dataset import create_dataloaders
from src.vae.model import HarmonicVAE

logger = logging.getLogger(__name__)


class HarmonicClusterAnalyzer:
    """Minimal analyzer for loading VAE, test features, and latent vectors."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        output_dir: str,
        normalizer_path: str = None,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer_path = normalizer_path

        self.cfg = OmegaConf.load(str(self.config_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> HarmonicVAE:
        checkpoint = torch.load(self.model_path, map_location=self.device)

        model_cfg = self.cfg.model
        model_params: Dict[str, Any] = {
            "input_dim": model_cfg.input_dim,
            "latent_dim": model_cfg.latent_dim,
            "hidden_dims": list(model_cfg.hidden_dims),
        }
        if hasattr(model_cfg, "loss"):
            model_params["loss"] = OmegaConf.to_container(model_cfg.loss, resolve=True)

        model = HarmonicVAE(**model_params).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def load_test_data(self, inference_data_cfg=None) -> Tuple[np.ndarray, np.ndarray]:
        data_cfg = self.cfg.data if hasattr(self.cfg, "data") else self.cfg.dataset

        normalization_cfg = None
        if hasattr(data_cfg, "normalization"):
            normalization_cfg = OmegaConf.to_container(data_cfg.normalization, resolve=True)

        train_data_dir = str(data_cfg.train_data_dir)
        valid_data_dir = str(data_cfg.valid_data_dir)
        test_data_dir = str(data_cfg.test_data_dir)

        if inference_data_cfg is not None:
            train_data_dir = str(inference_data_cfg.get("train_data_dir", train_data_dir))
            valid_data_dir = str(inference_data_cfg.get("valid_data_dir", valid_data_dir))
            test_data_dir = str(inference_data_cfg.get("test_data_dir", test_data_dir))

        logger.info(
            "Clustering data dirs: train=%s valid=%s test=%s",
            train_data_dir,
            valid_data_dir,
            test_data_dir,
        )

        test_loader, _ = create_dataloaders(
            train_data_dir=train_data_dir,
            valid_data_dir=valid_data_dir,
            test_data_dir=test_data_dir,
            mode="test_only",
            feature_type=str(data_cfg.feature_type),
            batch_size=int(data_cfg.batch_size),
            num_workers=4,
            max_samples=data_cfg.get("max_samples", None),
            test_max_samples=data_cfg.get("test_max_samples", None),
            cache_features=bool(data_cfg.get("cache_features", True)),
            verify_data=bool(data_cfg.get("verify_data", True)),
            normalization=normalization_cfg,
            eager_cache=bool(data_cfg.get("eager_cache", False)),
        )

        features_list = []
        labels_list = []
        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
                    features, labels = batch_data[0], batch_data[1]
                else:
                    features = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
                    labels = torch.zeros(features.shape[0])

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy() if torch.is_tensor(labels) else labels)

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return features, labels

    def extract_latent_representations(self, features: np.ndarray) -> np.ndarray:
        data_cfg = self.cfg.data if hasattr(self.cfg, "data") else self.cfg.dataset
        batch_size = int(data_cfg.batch_size)

        latents = []
        n_batches = (len(features) + batch_size - 1) // batch_size

        self.model.eval()
        with torch.no_grad():
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(features))
                batch = torch.tensor(features[start:end], dtype=torch.float32).to(self.device)
                mu, _ = self.model.encode(batch)
                latents.append(mu.cpu().numpy())

        return np.concatenate(latents, axis=0)


def _build_cache_prefix(model_path: str, config_path: str, k: int, random_state: int) -> str:
    key = f"{model_path}|{config_path}|{k}|{random_state}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()[:10]
    return f"k{k}_seed{random_state}_{digest}"


def load_or_create_clustering_data(analyzer: HarmonicClusterAnalyzer, output_dir: str, cfg):
    cluster_cfg = cfg.clustering
    k = int(cluster_cfg.k)
    random_state = int(cluster_cfg.random_state)
    n_init = int(cluster_cfg.n_init)
    force_recalc = bool(cluster_cfg.get("force_recalc", False))

    cluster_cache_dir = Path(output_dir) / "cluster_cache"
    prefix = _build_cache_prefix(str(analyzer.model_path), str(analyzer.config_path), k, random_state)

    cache_files = {
        "scaler": cluster_cache_dir / f"{prefix}_scaler.pkl",
        "kmeans": cluster_cache_dir / f"{prefix}_kmeans.pkl",
        "members": cluster_cache_dir / f"{prefix}_members.pkl",
        "centers": cluster_cache_dir / f"{prefix}_centers.pkl",
    }

    all_cache_exists = all(path.exists() for path in cache_files.values())
    if all_cache_exists and not force_recalc:
        with open(cache_files["scaler"], "rb") as f:
            scaler = pickle.load(f)
        with open(cache_files["kmeans"], "rb") as f:
            kmeans = pickle.load(f)
        with open(cache_files["members"], "rb") as f:
            members_by_cluster = pickle.load(f)
        with open(cache_files["centers"], "rb") as f:
            centers = pickle.load(f)
        return scaler, kmeans, members_by_cluster, centers

    harmonic_features, _ = analyzer.load_test_data(cfg.data)
    latent_features = analyzer.extract_latent_representations(harmonic_features)

    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_features)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    cluster_ids = kmeans.fit_predict(latent_scaled)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    members_by_cluster = [latent_features[cluster_ids == cid] for cid in range(k)]

    cluster_cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_files["scaler"], "wb") as f:
        pickle.dump(scaler, f)
    with open(cache_files["kmeans"], "wb") as f:
        pickle.dump(kmeans, f)
    with open(cache_files["members"], "wb") as f:
        pickle.dump(members_by_cluster, f)
    with open(cache_files["centers"], "wb") as f:
        pickle.dump(centers, f)

    return scaler, kmeans, members_by_cluster, centers
