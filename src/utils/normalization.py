import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DDSPNormalizer:
    """DDSPスタイル正規化の可逆実装"""
    
    def __init__(
        self,
        exponent: float = 10.0,
        max_value: float = 2.0,
        threshold: float = 1e-7,
        eps: float = 1e-8
    ):
        self.exponent = exponent
        self.max_value = max_value
        self.threshold = threshold
        self.eps = eps
        
    def exp_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """DDSPのexp_sigmoid関数"""
        sigmoid_out = 1.0 / (1.0 + np.exp(-x))
        return self.max_value * (sigmoid_out ** np.log(self.exponent)) + self.threshold
    
    def inv_exp_sigmoid(self, y: np.ndarray) -> np.ndarray:
        """exp_sigmoidの逆関数"""
        # y = max * (sig(x)**log(e)) + th
        # ⇒ sig(x) = ((y-th)/max)**(1/log(e))
        s = ((y - self.threshold) / self.max_value)**(1.0 / np.log(self.exponent))
        # numerical clamp
        s = np.clip(s, self.eps, 1.0 - self.eps)
        # x = logit(s)
        return np.log(s / (1.0 - s))
    
    def normalize(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ロジット → 正規化済み分布, かけ戻し用の合計
        Args:
            logits: shape [..., n_harmonics]
        Returns:
            normalized: [..., n_harmonics] （和が1.0）
            sum_before: [..., 1] 正規化する前の合計
        """
        y = self.exp_sigmoid(logits)
        sum_before = np.sum(y, axis=-1, keepdims=True)
        normalized = y / (sum_before + self.eps)
        return normalized, sum_before
    
    def denormalize(self, normalized: np.ndarray, sum_before: np.ndarray) -> np.ndarray:
        """
        正規化済み分布 + 合計 → ロジット
        """
        # 1) 正規化前の振幅に戻す
        y = normalized * sum_before
        # 2) exp_sigmoidの逆変換でlogitsに戻す
        return self.inv_exp_sigmoid(y)

class DataNormalizer:
    """データ正規化のためのクラス"""
    
    def __init__(
        self,
        method: str = "standardize",  # "standardize", "minmax", "robust", "ddsp", "none"
        feature_range: Tuple[float, float] = (0.0, 1.0),
        epsilon: float = 1e-8,
        clip_outliers: bool = False,
        outlier_percentiles: Tuple[float, float] = (1.0, 99.0)
    ):
        """
        Args:
            method: 正規化手法 ("standardize", "minmax", "robust", "ddsp", "none")
            feature_range: MinMaxスケーリング時の範囲
            epsilon: ゼロ除算防止のための小さな値
            clip_outliers: 外れ値をクリップするかどうか
            outlier_percentiles: 外れ値クリップの閾値パーセンタイル
        """
        self.method = method
        # ListConfigをtupleに変換
        if hasattr(feature_range, '__iter__') and not isinstance(feature_range, (str, tuple)):
            self.feature_range = tuple(feature_range)
        else:
            self.feature_range = feature_range
            
        if hasattr(outlier_percentiles, '__iter__') and not isinstance(outlier_percentiles, (str, tuple)):
            self.outlier_percentiles = tuple(outlier_percentiles)
        else:
            self.outlier_percentiles = outlier_percentiles
            
        self.epsilon = epsilon
        self.clip_outliers = clip_outliers
        
        # 統計値を保存
        self.stats = {}
        self.is_fitted = False
        
        # DDSPスタイル正規化用のオブジェクト
        if self.method == "ddsp":
            self.ddsp_normalizer = DDSPNormalizer()
            # DDSPでは合計値も保存する必要がある
            self.sum_before_cache = {}
        
    def fit(self, data: np.ndarray) -> 'DataNormalizer':
        """
        データから正規化の統計値を計算
        
        Args:
            data: 学習データ [n_samples, n_features]
            
        Returns:
            self (method chaining用)
        """
        if self.method == "none":
            self.is_fitted = True
            return self
            
        logger.info(f"Fitting normalizer with method: {self.method}")
        logger.info(f"Data shape: {data.shape}")
        
        if self.method == "standardize":
            # Z-score正規化 (平均0, 標準偏差1)
            self.stats['mean'] = np.mean(data, axis=0)
            self.stats['std'] = np.std(data, axis=0)
            # 標準偏差が0の特徴量を処理
            self.stats['std'] = np.where(self.stats['std'] < self.epsilon, 1.0, self.stats['std'])
            
        elif self.method == "minmax":
            # Min-Max正規化
            self.stats['min'] = np.min(data, axis=0)
            self.stats['max'] = np.max(data, axis=0)
            # 範囲が0の特徴量を処理
            data_range = self.stats['max'] - self.stats['min']
            self.stats['range'] = np.where(data_range < self.epsilon, 1.0, data_range)
            
        elif self.method == "robust":
            # ロバスト正規化 (中央値とIQR使用)
            self.stats['median'] = np.median(data, axis=0)
            self.stats['q25'] = np.percentile(data, 25, axis=0)
            self.stats['q75'] = np.percentile(data, 75, axis=0)
            iqr = self.stats['q75'] - self.stats['q25']
            self.stats['iqr'] = np.where(iqr < self.epsilon, 1.0, iqr)
            
        elif self.method == "ddsp":
            # DDSPスタイル正規化用（統計値は不要だが、一貫性のために記録）
            self.stats['method'] = 'ddsp'
            # 学習データの統計をサンプリングして保存（参考用）
            sample_size = min(1000, len(data))
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            sample_data = data[sample_indices]
            
            # サンプルデータでDDSP正規化を試行
            normalized_samples = []
            sum_before_samples = []
            
            for i in range(len(sample_data)):
                normalized, sum_before = self.ddsp_normalizer.normalize(sample_data[i:i+1])
                normalized_samples.append(normalized)
                sum_before_samples.append(sum_before)
            
            # 統計情報を保存（参考用）
            all_normalized = np.concatenate(normalized_samples, axis=0)
            all_sum_before = np.concatenate(sum_before_samples, axis=0)
            
            self.stats['normalized_mean'] = np.mean(all_normalized, axis=0)
            self.stats['normalized_std'] = np.std(all_normalized, axis=0)
            self.stats['sum_before_mean'] = np.mean(all_sum_before)
            self.stats['sum_before_std'] = np.std(all_sum_before)
            
            logger.info("DDSP-style normalization fitted with sample statistics")
        
        # 外れ値クリップ用の統計値
        if self.clip_outliers:
            self.stats['clip_min'] = np.percentile(data, self.outlier_percentiles[0], axis=0)
            self.stats['clip_max'] = np.percentile(data, self.outlier_percentiles[1], axis=0)
        
        self.is_fitted = True
        
        # 統計値をログ出力
        self._log_stats()
        
        return self
    
    def transform(self, data: np.ndarray, return_sum_before: bool = False) -> np.ndarray:
        """
        データを正規化
        
        Args:
            data: 正規化対象データ [n_samples, n_features]
            return_sum_before: DDSPの場合、合計値も返すかどうか
            
        Returns:
            正規化されたデータ (DDSPの場合、return_sum_before=Trueなら(normalized, sum_before)のタプル)
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
            
        if self.method == "none":
            return data.copy()
        elif self.method == "ddsp":
            # DDSPスタイル正規化：生のlogitsをharmonic distributionに変換
            normalized, sum_before = self.ddsp_normalizer.normalize(data)
            
            # 合計値をキャッシュ（逆変換用）
            if hasattr(self, 'sum_before_cache'):
                # データのハッシュをキーとして使用（簡単な実装）
                data_hash = hash(data.tobytes())
                self.sum_before_cache[data_hash] = sum_before
            
            if return_sum_before:
                return normalized, sum_before
            else:
                return normalized
        
        data_normalized = data.copy()
        
        # 外れ値クリップ
        if self.clip_outliers:
            data_normalized = np.clip(
                data_normalized,
                self.stats['clip_min'],
                self.stats['clip_max']
            )
        
        # 正規化実行
        if self.method == "standardize":
            data_normalized = (data_normalized - self.stats['mean']) / self.stats['std']
            
        elif self.method == "minmax":
            data_normalized = (data_normalized - self.stats['min']) / self.stats['range']
            # 指定範囲にスケール
            data_normalized = (data_normalized * (self.feature_range[1] - self.feature_range[0]) + 
                             self.feature_range[0])
            
        elif self.method == "robust":
            data_normalized = (data_normalized - self.stats['median']) / self.stats['iqr']
        
        return data_normalized
    
    def inverse_transform(self, data: np.ndarray, sum_before: Optional[np.ndarray] = None) -> np.ndarray:
        """
        正規化を逆変換
        
        Args:
            data: 正規化されたデータ
            sum_before: DDSPの場合、正規化前の合計値
            
        Returns:
            元のスケールに戻されたデータ
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
            
        if self.method == "none":
            return data.copy()
        elif self.method == "ddsp":
            # DDSPスタイル正規化の逆変換
            if sum_before is None:
                # キャッシュから合計値を取得を試みる
                data_hash = hash(data.tobytes())
                if hasattr(self, 'sum_before_cache') and data_hash in self.sum_before_cache:
                    sum_before = self.sum_before_cache[data_hash]
                else:
                    logger.warning("DDSP inverse transform: sum_before not provided and not found in cache. "
                                 "Using estimated values from training statistics.")
                    # 学習時の統計値を使用してsum_beforeを推定
                    sum_before = np.full((data.shape[0], 1), self.stats.get('sum_before_mean', 1.0))
            
            return self.ddsp_normalizer.denormalize(data, sum_before)
        
        data_original = data.copy()
        
        if self.method == "standardize":
            data_original = data_original * self.stats['std'] + self.stats['mean']
            
        elif self.method == "minmax":
            # 指定範囲から[0,1]に戻す
            data_original = ((data_original - self.feature_range[0]) / 
                           (self.feature_range[1] - self.feature_range[0]))
            # 元のスケールに戻す
            data_original = data_original * self.stats['range'] + self.stats['min']
            
        elif self.method == "robust":
            data_original = data_original * self.stats['iqr'] + self.stats['median']
        
        return data_original
    
    def fit_transform(self, data: np.ndarray, return_sum_before: bool = False) -> np.ndarray:
        """
        統計値計算と正規化を同時実行
        
        Args:
            data: 学習データ
            return_sum_before: DDSPの場合、合計値も返すかどうか
            
        Returns:
            正規化されたデータ
        """
        self.fit(data)
        return self.transform(data, return_sum_before=return_sum_before)
    
    def save(self, filepath: str) -> None:
        """
        正規化器の状態を保存
        
        Args:
            filepath: 保存先パス
        """
        def convert_to_serializable(obj):
            """オブジェクトをJSON serializable な形式に変換"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):  # NumPy float types
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):  # NumPy int types
                return int(obj)
            elif isinstance(obj, np.bool_):  # NumPy bool type
                return bool(obj)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, dict)):
                # ListConfig等のiterableをリストに変換
                return list(obj)
            elif hasattr(obj, '_content'):  # OmegaConf objects
                return obj._content if hasattr(obj, '_content') else str(obj)
            else:
                return obj
        
        save_data = {
            'method': self.method,
            'feature_range': convert_to_serializable(self.feature_range),
            'epsilon': float(self.epsilon),
            'clip_outliers': bool(self.clip_outliers),
            'outlier_percentiles': convert_to_serializable(self.outlier_percentiles),
            'stats': {k: convert_to_serializable(v) for k, v in self.stats.items()},
            'is_fitted': bool(self.is_fitted)
        }
        
        # DDSPパラメータも保存
        if self.method == "ddsp":
            save_data['ddsp_params'] = {
                'exponent': float(self.ddsp_normalizer.exponent),
                'max_value': float(self.ddsp_normalizer.max_value),
                'threshold': float(self.ddsp_normalizer.threshold),
                'eps': float(self.ddsp_normalizer.eps)
            }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Normalizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataNormalizer':
        """
        保存された正規化器を読み込み
        
        Args:
            filepath: 読み込み元パス
            
        Returns:
            DataNormalizerインスタンス
        """
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        normalizer = cls(
            method=save_data['method'],
            feature_range=tuple(save_data['feature_range']),
            epsilon=save_data['epsilon'],
            clip_outliers=save_data['clip_outliers'],
            outlier_percentiles=tuple(save_data['outlier_percentiles'])
        )
        
        # 統計値を復元
        normalizer.stats = {k: np.array(v) if isinstance(v, list) else v 
                           for k, v in save_data['stats'].items()}
        normalizer.is_fitted = save_data['is_fitted']
        
        # DDSPパラメータも復元
        if normalizer.method == "ddsp":
            if 'ddsp_params' in save_data:
                ddsp_params = save_data['ddsp_params']
                normalizer.ddsp_normalizer = DDSPNormalizer(
                    exponent=ddsp_params['exponent'],
                    max_value=ddsp_params['max_value'],
                    threshold=ddsp_params['threshold'],
                    eps=ddsp_params['eps']
                )
            else:
                normalizer.ddsp_normalizer = DDSPNormalizer()
            normalizer.sum_before_cache = {}
        
        logger.info(f"Normalizer loaded from {filepath}")
        return normalizer
    
    def _log_stats(self) -> None:
        """統計値をログ出力"""
        logger.info(f"Normalization statistics ({self.method}):")
        
        if self.method == "standardize":
            logger.info(f"  Mean range: [{np.min(self.stats['mean']):.4f}, {np.max(self.stats['mean']):.4f}]")
            logger.info(f"  Std range: [{np.min(self.stats['std']):.4f}, {np.max(self.stats['std']):.4f}]")
            
        elif self.method == "minmax":
            logger.info(f"  Min range: [{np.min(self.stats['min']):.4f}, {np.max(self.stats['min']):.4f}]")
            logger.info(f"  Max range: [{np.min(self.stats['max']):.4f}, {np.max(self.stats['max']):.4f}]")
            
        elif self.method == "robust":
            logger.info(f"  Median range: [{np.min(self.stats['median']):.4f}, {np.max(self.stats['median']):.4f}]")
            logger.info(f"  IQR range: [{np.min(self.stats['iqr']):.4f}, {np.max(self.stats['iqr']):.4f}]")
            
        elif self.method == "ddsp":
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
    method: str = "standardize",
    max_samples: int = None,
    **kwargs
) -> DataNormalizer:
    """
    データローダーから正規化器を作成
    
    Args:
        train_loader: 学習データローダー
        method: 正規化手法
        max_samples: 統計計算に使用する最大サンプル数（Noneで全データ使用）
        **kwargs: DataNormalizerの追加引数
        
    Returns:
        フィットされたDataNormalizer
    """
    logger.info("Creating normalizer from training dataset...")
    
    all_features = []
    sample_count = 0
    
    for batch_data in train_loader:
        # データローダーの戻り値を適切に処理
        if isinstance(batch_data, (tuple, list)):
            if len(batch_data) == 2:
                features, _ = batch_data  # (features, labels)
            else:
                features = batch_data[0]
        else:
            features = batch_data
        
        # numpy配列に変換
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        all_features.append(features)
        sample_count += features.shape[0]
        
        # max_samplesが指定されている場合のみ制限をチェック
        if max_samples is not None and sample_count >= max_samples:
            break
    
    # 全特徴量を結合
    all_features = np.concatenate(all_features, axis=0)
    if max_samples is not None:
        all_features = all_features[:max_samples]
    
    logger.info(f"Using {all_features.shape[0]} samples for normalization statistics")
    
    # 正規化器を作成してフィット
    normalizer = DataNormalizer(method=method, **kwargs)
    normalizer.fit(all_features)
    
    return normalizer

# PyTorchテンソル用のヘルパー関数
def normalize_tensor(
    tensor: torch.Tensor,
    normalizer: DataNormalizer
) -> torch.Tensor:
    """
    PyTorchテンソルを正規化
    
    Args:
        tensor: 正規化対象テンソル
        normalizer: フィットされた正規化器
        
    Returns:
        正規化されたテンソル
    """
    numpy_data = tensor.cpu().numpy()
    normalized_data = normalizer.transform(numpy_data)
    return torch.from_numpy(normalized_data).to(tensor.device)

def denormalize_tensor(
    tensor: torch.Tensor,
    normalizer: DataNormalizer,
    sum_before: Optional[np.ndarray] = None
) -> torch.Tensor:
    """
    PyTorchテンソルの正規化を逆変換
    
    Args:
        tensor: 正規化されたテンソル
        normalizer: フィットされた正規化器
        sum_before: DDSPの場合、正規化前の合計値
        
    Returns:
        元のスケールに戻されたテンソル
    """
    numpy_data = tensor.cpu().numpy()
    denormalized_data = normalizer.inverse_transform(numpy_data, sum_before)
    return torch.from_numpy(denormalized_data).to(tensor.device)