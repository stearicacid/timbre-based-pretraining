import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
import logging
import os
import json
from pathlib import Path
import glob
import re 
from src.utils.normalization import DataNormalizer, create_normalizer_from_dataset

logger = logging.getLogger(__name__)

class HarmonicDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        feature_type: str = 'harmonic',
        max_samples: Optional[int] = None,
        train_max_samples: Optional[int] = None,  
        valid_max_samples: Optional[int] = None,  
        test_max_samples: Optional[int] = None, 
        cache_features: bool = True,
        verify_data: bool = True,
        normalizer: Optional[DataNormalizer] = None,
        pool_time_dim: bool = True,  
        eager_cache: bool = False   
    ):
        """
        Dataset for harmonic features from prepare_dataset.py output.
        
        Args:
            data_dir: Base directory containing feature files
            mode: 'train', 'valid', or 'test'
            feature_type: Type of features to load ('harmonic' or 'spectrogram')
            max_samples: Maximum number of samples to load (None for all)
            cache_features: Whether to cache features in memory
            verify_data: Whether to verify data integrity
            pool_time_dim: Whether to apply mean pooling over the time dimension
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.feature_type = feature_type
        self.max_samples = max_samples
        self.train_max_samples = train_max_samples
        self.valid_max_samples = valid_max_samples
        self.test_max_samples = test_max_samples
        self.cache_features = cache_features
        self.verify_data = verify_data
        self.pool_time_dim = pool_time_dim  # フラグをインスタンス変数として保存
        self.eager_cache = eager_cache      # 追加
        
        # Cache for features and labels
        self.features_cache = {}
        self.labels_cache = {}
        self.metadata_cache = {}
        
        self.normalizer = normalizer
        
        logger.info(f"Initializing {mode} dataset from {self.data_dir}")

        self._load_file_paths()
        
        if self.verify_data:
            self._verify_dataset()
        
        # 追加: 初回に全件をキャッシュ（メモリ余裕がある場合のみ推奨）
        if self.cache_features and self.eager_cache:
            self._eager_fill_cache()

        logger.info(f"{mode} dataset initialized with {len(self.feature_files)} samples")

    # 追加: 全件キャッシュ関数
    def _eager_fill_cache(self):
        logger.info("Eager caching all features/labels into CPU memory...")
        for idx in range(len(self.feature_files)):
            # _load_feature/_load_label は cache_features=True の場合キャッシュに載る
            _ = self._load_feature(idx)
            _ = self._load_label(idx)
        logger.info("Eager cache completed")

    def _load_file_paths(self):
        """Load feature and label file paths with natural sorting."""
        # Look for feature files
        feature_pattern = f"feature_*.npy"
        # globでファイルリストをまず取得
        feature_files_unsorted = glob.glob(str(self.data_dir / feature_pattern))

        # 自然順ソートのためのキー関数を定義
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        # キー関数を使ってファイルリストをソート
        self.feature_files = sorted(feature_files_unsorted, key=natural_sort_key)
        
        if not self.feature_files:
            raise FileNotFoundError(
                f"No feature files found in {self.data_dir} with pattern {feature_pattern}"
            )
        
        # Extract file IDs and create corresponding paths
        self.file_ids = []
        self.label_files = []
        self.metadata_files = []
        
        for feature_file in self.feature_files:
            # Extract file ID from feature file name
            # e.g., "feature_tfds_train_0.npy" -> "tfds_train_0"
            filename = Path(feature_file).stem
            file_id = filename.replace('feature_', '')
            self.file_ids.append(file_id)
            
            # Create corresponding label and metadata paths
            label_file = self.data_dir / f"label_{file_id}.npy"
            metadata_file = self.data_dir / f"metadata_{file_id}.json"
            
            self.label_files.append(str(label_file))
            self.metadata_files.append(str(metadata_file))
        
        # Limit samples if specified
        if self.max_samples is not None:
            self.feature_files = self.feature_files[:self.max_samples]
            self.label_files = self.label_files[:self.max_samples]
            self.metadata_files = self.metadata_files[:self.max_samples]
            self.file_ids = self.file_ids[:self.max_samples]
        
        logger.info(f"Found {len(self.feature_files)} feature files")
    
    def _verify_dataset(self):
        """Verify dataset integrity."""
        logger.info("Verifying dataset integrity...")
        
        missing_labels = []
        missing_metadata = []
        invalid_features = []
        
        for i, (feature_file, label_file, metadata_file) in enumerate(
            zip(self.feature_files, self.label_files, self.metadata_files)
        ):
            # Check if label file exists
            if not os.path.exists(label_file):
                missing_labels.append(label_file)
            
            # Check if metadata file exists
            if not os.path.exists(metadata_file):
                missing_metadata.append(metadata_file)
            
            # Check feature file integrity (sample a few)
            if i % 100 == 0:  # Check every 100th file
                try:
                    features = np.load(feature_file)
                    if self.feature_type == 'harmonic':
                        # For harmonic features, expect shape (time_steps, n_harmonics)
                        if len(features.shape) != 2 or features.shape[1] != 45:
                            invalid_features.append(f"{feature_file}: shape {features.shape}")
                    elif self.feature_type == 'spectrogram':
                        # For spectrogram features, expect 2D array
                        if len(features.shape) != 2:
                            invalid_features.append(f"{feature_file}: shape {features.shape}")
                except Exception as e:
                    invalid_features.append(f"{feature_file}: {str(e)}")
        
        # Report issues
        if missing_labels:
            logger.warning(f"Missing {len(missing_labels)} label files")
        if missing_metadata:
            logger.warning(f"Missing {len(missing_metadata)} metadata files")
        if invalid_features:
            logger.warning(f"Invalid features found: {invalid_features[:5]}...")  # Show first 5
        
        if missing_labels or invalid_features:
            logger.warning("Dataset verification found issues, but continuing...")
    
    def _load_feature(self, idx: int) -> np.ndarray:
        """Load and process feature file."""
        # キャッシュキーに時間平均化フラグを含める
        cache_key = (idx, self.pool_time_dim)
        if self.cache_features and cache_key in self.features_cache:
            return self.features_cache[cache_key]
        
        feature_file = self.feature_files[idx]
        
        try:
            features = np.load(feature_file)
            
            # Process features based on type
            if self.feature_type == 'harmonic':
                # Features shape: (time_steps, n_harmonics) or (n_harmonics,)
                if self.pool_time_dim and len(features.shape) == 2:
                    # 前半の500フレーム（2秒分）のみを使用
                    features = features[:500]  # 追加行
                    # 時間平均化フラグがTrueの場合のみ平均化
                    features = np.mean(features, axis=0)
                elif len(features.shape) == 1 and not self.pool_time_dim:
                    # 時間次元がないが、時間次元付きが要求された場合
                    logger.warning(f"Feature file {feature_file} has no time dimension. Returning as is.")
                elif len(features.shape) > 2:
                    raise ValueError(f"Unexpected harmonic feature shape: {features.shape}")
                
                # Ensure we have 45 harmonics
                n_harmonics_dim = -1
                if features.shape[n_harmonics_dim] != 45:
                    logger.warning(f"Expected 45 harmonics, got {features.shape[n_harmonics_dim]} in {feature_file}")
                    # Pad or truncate to 45
                    pad_width = [(0, 0)] * features.ndim
                    if features.shape[n_harmonics_dim] < 45:
                        pad_width[n_harmonics_dim] = (0, 45 - features.shape[n_harmonics_dim])
                        features = np.pad(features, pad_width, 'constant')
                    else:
                        slicer = [slice(None)] * features.ndim
                        slicer[n_harmonics_dim] = slice(0, 45)
                        features = features[tuple(slicer)]
            
            elif self.feature_type == 'spectrogram':
                # For spectrogram, keep as is or apply temporal pooling
                if self.pool_time_dim and len(features.shape) == 2:
                    # Apply average pooling over time if needed
                    features = np.mean(features, axis=0)
            
            # Convert to float32 BEFORE normalization
            features = features.astype(np.float32)

            # Apply DDSP normalizer when enabled.
            if self.normalizer is not None:
                features = self.normalizer.transform(features)
                features = features.astype(np.float32)

            # Cache if enabled
            if self.cache_features:
                self.features_cache[cache_key] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error loading feature {feature_file}: {e}")
            # エラーを再発生させる
            raise RuntimeError(f"Failed to load feature file {feature_file}: {e}") from e
    
    def _load_label(self, idx: int) -> int:
        """Load label file."""
        if self.cache_features and idx in self.labels_cache:
            return self.labels_cache[idx]
        
        label_file = self.label_files[idx]
        
        try:
            if os.path.exists(label_file):
                label = np.load(label_file)
                if isinstance(label, np.ndarray):
                    label = label.item()
                label = int(label)
            else:
                # Try to extract from metadata
                metadata = self._load_metadata(idx)
                label = metadata.get('label', 0)
            
            # Cache if enabled
            if self.cache_features:
                self.labels_cache[idx] = label
            
            return label
            
        except Exception as e:
            logger.warning(f"Error loading label {label_file}: {e}")
            return 0  # Default label
    
    def _load_metadata(self, idx: int) -> dict:
        """Load metadata file."""
        if self.cache_features and idx in self.metadata_cache:
            return self.metadata_cache[idx]
        
        metadata_file = self.metadata_files[idx]
        
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Cache if enabled
            if self.cache_features:
                self.metadata_cache[idx] = metadata
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error loading metadata {metadata_file}: {e}")
            return {}
    
    def __len__(self) -> int:
        return len(self.feature_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        features = self._load_feature(idx)
        label = self._load_label(idx)
        
        # __getitem__はDataLoaderで使われるため、Tensorを返す必要がある
        # 時間次元を持つデータはcollatingが難しくなるため、ここでは時間平均化されたものを返す
        # 時間次元付きデータはget_sample_info経由で取得することを推奨
        if self.pool_time_dim is False and len(features.shape) > 1:
             features = np.mean(features, axis=0)

        return torch.FloatTensor(features), torch.LongTensor([label])
    
    def get_sample_info(self, idx: int) -> dict:
        """Get detailed information about a sample."""
        metadata = self._load_metadata(idx)
        features = self._load_feature(idx)
        label = self._load_label(idx)
        
        return {
            'idx': idx,
            'file_id': self.file_ids[idx],
            'feature': features,  # 特徴量データを追加
            'feature_shape': features.shape,
            'label': label,
            'metadata': metadata
        }

def create_dataloaders(
    train_data_dir: str,
    valid_data_dir: str,
    test_data_dir: str,
    mode: str = 'train_val',  # 'train_val', 'train_only', 'test_only'
    feature_type: str = 'harmonic',
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_samples: Optional[int] = None, 
    train_max_samples: Optional[int] = None,  
    valid_max_samples: Optional[int] = None,  
    test_max_samples: Optional[int] = None,   
    cache_features: bool = True,
    verify_data: bool = True,
    normalization: Optional[dict] = None,
    prefetch_factor: int = 2,          # 追加
    persistent_workers: bool = True,   # 追加
    eager_cache: bool = False,         # 追加
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from prepared dataset.
    
    Args:
        train_data_dir: Path to training data directory
        valid_data_dir: Path to validation data directory  
        test_data_dir: Path to test data directory
        mode: Which datasets to load
        feature_type: Type of features to load
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory
        max_samples: Maximum samples per dataset (backward compatibility)
        train_max_samples: Maximum training samples
        valid_max_samples: Maximum validation samples
        test_max_samples: Maximum test samples
        cache_features: Cache features in memory
        verify_data: Verify data integrity
        normalization: Normalization configuration dictionary
        
    Returns:
        train_loader, val_loader
    """
    
    # 個別のmax_samplesが指定されていない場合は、共通のmax_samplesを使用
    actual_train_max = train_max_samples if train_max_samples is not None else max_samples
    actual_valid_max = valid_max_samples if valid_max_samples is not None else max_samples
    actual_test_max = test_max_samples if test_max_samples is not None else max_samples
    
    logger.info(f"Dataset sample limits - Train: {actual_train_max}, Valid: {actual_valid_max}, Test: {actual_test_max}")
    
    # 正規化器の準備
    normalizer = None
    if normalization and normalization.get('enabled', False):
        logger.info("Setting up data normalization...")
        
        max_samples_for_norm = normalization.get('max_samples', None)
        
        # 統計計算用のサンプル数を決定（訓練データから）
        if max_samples_for_norm is None:
            temp_max_samples = actual_train_max
        else:
            if actual_train_max is not None:
                temp_max_samples = min(max_samples_for_norm, actual_train_max)
            else:
                temp_max_samples = max_samples_for_norm
        
        logger.info(f"Using {temp_max_samples} training samples for normalization statistics")
        
        # まず正規化なしでトレーニングデータセットを作成（統計計算用）
        temp_train_dataset = HarmonicDataset(
            data_dir=train_data_dir,
            mode='train',
            feature_type=feature_type,
            max_samples=temp_max_samples,
            cache_features=False,
            verify_data=False,
            normalizer=None
        )
        
        temp_train_loader = DataLoader(
            temp_train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # DDSP正規化器設定をConfigから取得
        normalizer_kwargs = {
            'exponent': normalization.get('exponent', 10.0),
            'max_value': normalization.get('max_value', 2.0),
            'threshold': normalization.get('threshold', 1e-7),
            'eps': normalization.get('eps', 1e-8),
        }
        
        # 正規化器を作成してフィット
        normalizer = create_normalizer_from_dataset(
            temp_train_loader,
            max_samples=None,
            **normalizer_kwargs
        )
        logger.info(f"Normalization config: {normalizer_kwargs}")
    
    loaders = []
    
    if mode in ['train_val', 'train_only']:
        # Create training dataset
        logger.info(f"Loading training data from {train_data_dir} (max_samples: {actual_train_max})")
        train_dataset = HarmonicDataset(
            data_dir=train_data_dir,
            mode='train',
            feature_type=feature_type,
            max_samples=actual_train_max,  # 個別指定されたサンプル数を使用
            cache_features=cache_features,
            verify_data=verify_data,
            normalizer=normalizer,
            eager_cache=eager_cache
        )
        
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        if num_workers and num_workers > 0:
            loader_kwargs.update(dict(
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            ))
        train_loader = DataLoader(train_dataset, **loader_kwargs)
        loaders.append(train_loader)
        
        logger.info(f"Training dataset: {len(train_dataset)} samples, "
                   f"{len(train_loader)} batches")
    
    if mode in ['train_val', 'val_only']:
        # Create validation dataset
        logger.info(f"Loading validation data from {valid_data_dir} (max_samples: {actual_valid_max})")
        val_dataset = HarmonicDataset(
            data_dir=valid_data_dir,
            mode='valid',
            feature_type=feature_type,
            max_samples=actual_valid_max,  # 個別指定されたサンプル数を使用
            cache_features=cache_features,
            verify_data=verify_data,
            normalizer=normalizer,
            eager_cache=eager_cache
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        if num_workers and num_workers > 0:
            loader_kwargs.update(dict(
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            ))
        val_loader = DataLoader(val_dataset, **loader_kwargs)
        loaders.append(val_loader)
        
        logger.info(f"Validation dataset: {len(val_dataset)} samples, "
                   f"{len(val_loader)} batches")
    
    if mode == 'test_only':
        # Create test dataset
        logger.info(f"Loading test data from {test_data_dir} (max_samples: {actual_test_max})")
        test_dataset = HarmonicDataset(
            data_dir=test_data_dir,
            mode='test',
            feature_type=feature_type,
            max_samples=actual_test_max,  # 個別指定されたサンプル数を使用
            cache_features=cache_features,
            verify_data=verify_data,
            normalizer=normalizer,
            eager_cache=eager_cache
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        if num_workers and num_workers > 0:
            loader_kwargs.update(dict(
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            ))
        test_loader = DataLoader(test_dataset, **loader_kwargs)
        loaders.append(test_loader)
        
        logger.info(f"Test dataset: {len(test_dataset)} samples, "
                   f"{len(test_loader)} batches")
    
    # 正規化器を保存（設定されている場合）
    if normalizer is not None and normalization.get('save_normalizer', True):
        normalizer_path = f"normalizer_{feature_type}.json"
        normalizer.save(normalizer_path)
        logger.info(f"Normalizer saved to {normalizer_path}")
    
    # Return appropriate loaders based on mode
    if mode == 'train_val':
        return loaders[0], loaders[1]  # train_loader, val_loader
    elif mode == 'train_only':
        return loaders[0], None  # train_loader, None
    elif mode == 'val_only':
        return None, loaders[0]  # None, val_loader
    elif mode == 'test_only':
        return loaders[0], None  # test_loader, None
    else:
        raise ValueError(f"Invalid mode: {mode}")

# Utility functions
def get_dataset_stats(data_dir: str, feature_type: str = 'harmonic') -> dict:
    """Get statistics about the dataset."""
    dataset = HarmonicDataset(
        data_dir=data_dir,
        mode='analysis',
        feature_type=feature_type,
        cache_features=False,
        verify_data=False
    )
    
    if len(dataset) == 0:
        return {"num_samples": 0}
    
    # Sample a subset for statistics
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    features_list = []
    labels_list = []
    
    for idx in indices:
        features, label = dataset[idx]
        features_list.append(features.numpy())
        labels_list.append(label.item())
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    stats = {
        "num_samples": len(dataset),
        "feature_shape": features_array[0].shape,
        "feature_mean": np.mean(features_array, axis=0),
        "feature_std": np.std(features_array, axis=0),
        "feature_min": np.min(features_array, axis=0),
        "feature_max": np.max(features_array, axis=0),
        "unique_labels": np.unique(labels_array).tolist(),
        "label_counts": {int(label): int(count) for label, count in 
                        zip(*np.unique(labels_array, return_counts=True))}
    }
    
    return stats

def create_dataset_summary(train_dir: str, valid_dir: str, test_dir: str, 
                          feature_type: str = 'harmonic') -> dict:
    """Create a comprehensive dataset summary."""
    summary = {}
    
    for split, data_dir in [('train', train_dir), ('valid', valid_dir), ('test', test_dir)]:
        logger.info(f"Analyzing {split} dataset...")
        try:
            stats = get_dataset_stats(data_dir, feature_type)
            summary[split] = stats
        except Exception as e:
            logger.error(f"Error analyzing {split} dataset: {e}")
            summary[split] = {"error": str(e)}
    
    return summary