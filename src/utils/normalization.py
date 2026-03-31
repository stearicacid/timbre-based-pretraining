import torch
import numpy as np
from typing import Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

class DataNormalizer:

    def __init__(
        self,
        exponent: float = 10.0,
        max_value: float = 2.0,
        threshold: float = 1e-7,
        eps: float = 1e-8,
    ):
        self.exponent = exponent
        self.max_value = max_value
        self.threshold = threshold
        self.eps = eps

        self.stats = {}
        self.is_fitted = False
        self.sum_before_cache = {}
        
    def exp_sigmoid_np(self, x: np.ndarray) -> np.ndarray:
        sigmoid_out = 1.0 / (1.0 + np.exp(-x))
        return self.max_value * (sigmoid_out ** np.log(self.exponent)) + self.threshold
    
    def inv_exp_sigmoid_np(self, y: np.ndarray) -> np.ndarray:
        s = ((y - self.threshold) / self.max_value)**(1.0 / np.log(self.exponent))
        s = np.clip(s, self.eps, 1.0 - self.eps)
        return np.log(s / (1.0 - s))
    
    def normalize(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = self.exp_sigmoid_np(logits)
        sum_before = np.sum(y, axis=-1, keepdims=True)
        normalized = y / (sum_before + self.eps)
        return normalized, sum_before
    
    def denormalize(self, normalized: np.ndarray, sum_before: np.ndarray) -> np.ndarray:
        """
        正規化済み分布 + 合計 → ロジット
        """
        y = normalized * sum_before
        return self.inv_exp_sigmoid_np(y)        
        
    def fit(self, data: np.ndarray) -> 'DataNormalizer':
        logger.info(f"Data shape: {data.shape}")

        sample_size = min(1000, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[sample_indices]
            
        all_normalized, all_sum_before = self.normalize(sample_data)
        
        self.stats['normalized_mean'] = np.mean(all_normalized, axis=0)
        self.stats['normalized_std'] = np.std(all_normalized, axis=0)
        self.stats['sum_before_mean'] = np.mean(all_sum_before)
        self.stats['sum_before_std'] = np.std(all_sum_before)
        
        logger.info("Normalization fitted with sample statistics")
                
        self.is_fitted = True
        self._log_stats()
        
        return self
    
    def transform(self, data: np.ndarray, return_sum_before: bool = False) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
            
        normalized, sum_before = self.normalize(data)
         
        data_hash = hash(data.tobytes())
        self.sum_before_cache[data_hash] = sum_before
        
        if return_sum_before:
            return normalized, sum_before
        else:
            return normalized
    
    def inverse_transform(self, data: np.ndarray, sum_before: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        if sum_before is None:
            data_hash = hash(data.tobytes())
            if hasattr(self, 'sum_before_cache') and data_hash in self.sum_before_cache:
                sum_before = self.sum_before_cache[data_hash]
            else:
                logger.warning("DDSP inverse transform: sum_before not provided and not found in cache. "
                                "Using estimated values from training statistics.")
                sum_before = np.full((data.shape[0], 1), self.stats.get('sum_before_mean', 1.0))
        
        return self.denormalize(data, sum_before)

    
    def fit_transform(self, data: np.ndarray, return_sum_before: bool = False) -> np.ndarray:
        self.fit(data)
        return self.transform(data, return_sum_before=return_sum_before)
    
    def save(self, filepath: str) -> None:

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):  # NumPy float types
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):  # NumPy int types
                return int(obj)
            elif isinstance(obj, np.bool_):  # NumPy bool type
                return bool(obj)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, dict)):
                return list(obj)
            elif hasattr(obj, '_content'):  # OmegaConf objects
                return obj._content if hasattr(obj, '_content') else str(obj)
            else:
                return obj
        
        save_data = {
            'stats': {k: convert_to_serializable(v) for k, v in self.stats.items()},
            'is_fitted': bool(self.is_fitted)
        }
        
        save_data['ddsp_params'] = {
            'exponent': float(self.exponent),
            'max_value': float(self.max_value),
            'threshold': float(self.threshold),
            'eps': float(self.eps)
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Normalizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataNormalizer':
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        ddsp_params = save_data['ddsp_params']
        normalizer = cls(
            exponent=ddsp_params['exponent'],
            max_value=ddsp_params['max_value'],
            threshold=ddsp_params['threshold'],
            eps=ddsp_params['eps']
        )
        normalizer.stats = {k: np.array(v) if isinstance(v, list) else v 
                           for k, v in save_data['stats'].items()}
        normalizer.is_fitted = save_data['is_fitted']
        normalizer.sum_before_cache = {}
        
        logger.info(f"Normalizer loaded from {filepath}")
        return normalizer
    
    def _log_stats(self) -> None:
        logger.info("  DDSP-style normalization: exp_sigmoid + normalize_harmonics")
        logger.info(f"  Normalized mean range: [{np.min(self.stats['normalized_mean']):.4f}, {np.max(self.stats['normalized_mean']):.4f}]")
        logger.info(f"  Sum before mean: {self.stats['sum_before_mean']:.4f} ± {self.stats['sum_before_std']:.4f}")

# --- DDSP Style Normalization Functions (for backward compatibility) ---

def exp_sigmoid(x: torch.Tensor, exponent: float = 10.0, max_value: float = 2.0, threshold: float = 1e-7) -> torch.Tensor:
    """
    PyTorch implementation of DDSP's exp_sigmoid function.
    Bounds input to [threshold, max_value].
    """
    return max_value * torch.sigmoid(x) ** np.log(exponent) + threshold

def normalize_harmonics(harmonic_distribution: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    PyTorch implementation of DDSP's normalize_harmonics.
    Normalizes the last dimension to sum to 1.
    """
    sum_dist = torch.sum(harmonic_distribution, dim=-1, keepdim=True)
    return harmonic_distribution / (sum_dist + epsilon)

def ddsp_style_normalization(x: torch.Tensor) -> torch.Tensor:
    """
    Applies exp_sigmoid and then normalize_harmonics.
    This converts raw logits to a harmonic distribution.
    """
    x = exp_sigmoid(x)
    x = normalize_harmonics(x)
    return x

# --- End of DDSP Style Normalization Functions ---

def create_normalizer_from_dataset(
    train_loader: torch.utils.data.DataLoader,
    max_samples: int = None,
    **kwargs
) -> DataNormalizer:
    logger.info("Creating normalizer from training dataset...")
    
    all_features = []
    sample_count = 0
    
    for batch_data in train_loader:
        if isinstance(batch_data, (tuple, list)):
            if len(batch_data) == 2:
                features, _ = batch_data  # (features, labels)
            else:
                features = batch_data[0]
        else:
            features = batch_data
        
        # numpy
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        all_features.append(features)
        sample_count += features.shape[0]

        if max_samples is not None and sample_count >= max_samples:
            break

    all_features = np.concatenate(all_features, axis=0)
    if max_samples is not None:
        all_features = all_features[:max_samples]
    
    logger.info(f"Using {all_features.shape[0]} samples for normalization statistics")
    
    normalizer = DataNormalizer(**kwargs)
    normalizer.fit(all_features)
    
    return normalizer


def normalize_tensor(
    tensor: torch.Tensor,
    normalizer: DataNormalizer
) -> torch.Tensor:
    numpy_data = tensor.detach().cpu().numpy()
    normalized_data = normalizer.transform(numpy_data)
    return torch.from_numpy(normalized_data).to(device=tensor.device, dtype=tensor.dtype)

def denormalize_tensor(
    tensor: torch.Tensor,
    normalizer: DataNormalizer,
    sum_before: Optional[np.ndarray] = None
) -> torch.Tensor:
    numpy_data = tensor.detach().cpu().numpy()
    denormalized_data = normalizer.inverse_transform(numpy_data, sum_before)
    return torch.from_numpy(denormalized_data).to(device=tensor.device, dtype=tensor.dtype)