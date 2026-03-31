import numpy as np
import torch
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ClusteringMetrics:
    def __init__(self, n_clusters: Optional[int] = None):
        self.n_clusters = n_clusters
        
    def compute_clustering_metrics(
        self, 
        embeddings: np.ndarray, 
        true_labels: Union[List[str], np.ndarray, List[int]],
        compute_silhouette: bool = True,
        compute_ari: bool = True,
        compute_nmi: bool = True
    ) -> Dict[str, float]:
        """クラスタリング性能を計算"""
        metrics = {}
        
        if embeddings.size == 0 or len(true_labels) == 0:
            logger.warning("Empty embeddings or labels")
            return metrics
        
        # ラベルの型を統一（文字列のリストに変換）
        if isinstance(true_labels, np.ndarray):
            true_labels = true_labels.tolist()
        elif isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy().tolist()
        
        # 各要素を文字列に変換
        true_labels = [str(label) for label in true_labels]
        
        logger.info(f"Processing {len(true_labels)} labels with {embeddings.shape[0]} embeddings")
        logger.info(f"Sample labels: {true_labels[:5] if len(true_labels) >= 5 else true_labels}")
        
        # 真のラベルを数値化
        unique_labels = sorted(list(set(true_labels)))
        if len(unique_labels) <= 1:
            logger.warning(f"Only one unique label found: {unique_labels}")
            return metrics
        
        logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        true_labels_numeric = np.array([label_to_id[label] for label in true_labels])
        
        # クラスタ数の決定
        n_clusters = self.n_clusters or len(unique_labels)
        
        try:
            # 大量データの場合はサンプリング
            if embeddings.shape[0] > 10000:
                indices = np.random.choice(embeddings.shape[0], 10000, replace=False)
                sampled_embeddings = embeddings[indices]
                sampled_labels = true_labels_numeric[indices]
            else:
                sampled_embeddings = embeddings
                sampled_labels = true_labels_numeric
            
            # データの有効性チェック
            if sampled_embeddings.shape[0] < n_clusters:
                logger.warning(f"Not enough samples ({sampled_embeddings.shape[0]}) for {n_clusters} clusters")
                return metrics
                
            # K-meansクラスタリング
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(sampled_embeddings)
            
            # Silhouette Score
            if compute_silhouette and len(np.unique(predicted_labels)) > 1:
                silhouette = silhouette_score(sampled_embeddings, predicted_labels)
                metrics['silhouette_score'] = float(silhouette)
                
                # 真のラベルでのSilhouette Score
                if len(np.unique(sampled_labels)) > 1:
                    true_silhouette = silhouette_score(sampled_embeddings, sampled_labels)
                    metrics['true_silhouette_score'] = float(true_silhouette)
            
            # ARI (Adjusted Rand Index)
            if compute_ari:
                ari = adjusted_rand_score(sampled_labels, predicted_labels)
                metrics['ari'] = float(ari)
            
            # NMI (Normalized Mutual Information)
            if compute_nmi:
                nmi = normalized_mutual_info_score(sampled_labels, predicted_labels)
                metrics['nmi'] = float(nmi)
                
            logger.info(f"Computed clustering metrics: {metrics}")
                
        except Exception as e:
            logger.warning(f"Error computing clustering metrics: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            
        return metrics
    
    def compute_intra_inter_cluster_distances(
        self, 
        embeddings: np.ndarray, 
        true_labels: Union[List[str], np.ndarray, List[int]]
    ) -> Dict[str, float]:
        """クラスタ内距離とクラスタ間距離を計算"""
        if embeddings.size == 0 or len(true_labels) == 0:
            logger.warning("Empty embeddings or labels")
            return {}
        
        # ラベルの型を統一（文字列のリストに変換）
        if isinstance(true_labels, np.ndarray):
            true_labels = true_labels.tolist()
        elif isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy().tolist()
        
        # 各要素を文字列に変換
        true_labels = [str(label) for label in true_labels]
        
        unique_labels = sorted(list(set(true_labels)))
        if len(unique_labels) <= 1:
            logger.warning(f"Only one unique label found: {unique_labels}")
            return {}
        
        # クラスタ中心を計算
        cluster_centers = {}
        for label in unique_labels:
            # 文字列比較を使用してマスクを作成
            mask = np.array([l == label for l in true_labels])
            if np.any(mask):
                cluster_centers[label] = np.mean(embeddings[mask], axis=0)
        
        # クラスタ内距離（平均）
        intra_distances = []
        for label in unique_labels:
            mask = np.array([l == label for l in true_labels])
            if np.any(mask):
                cluster_points = embeddings[mask]
                if label in cluster_centers:
                    center = cluster_centers[label]
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    intra_distances.extend(distances)
        
        # クラスタ間距離（平均）
        inter_distances = []
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i < j and label1 in cluster_centers and label2 in cluster_centers:
                    dist = np.linalg.norm(cluster_centers[label1] - cluster_centers[label2])
                    inter_distances.append(dist)
        
        return {
            'mean_intra_distance': float(np.mean(intra_distances)) if intra_distances else 0.0,
            'mean_inter_distance': float(np.mean(inter_distances)) if inter_distances else 0.0,
            'separation_ratio': float(np.mean(inter_distances) / np.mean(intra_distances)) if intra_distances else 0.0
        }