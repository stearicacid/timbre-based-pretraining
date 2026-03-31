import torch
import numpy as np
from typing import Dict, List, Tuple
from .clustering_metrics import ClusteringMetrics
import logging

logger = logging.getLogger(__name__)

class TrainingMetrics:
    def __init__(self):
        self.clustering_metrics = ClusteringMetrics()
        
    def compute_comprehensive_metrics(
        self, 
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_samples: int = 5000
    ) -> Dict[str, float]:
        """包括的なメトリクスを計算"""
        model.eval()
        
        # データ収集
        embeddings = []
        labels = []
        recon_losses = []
        kl_divs = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # 最大サンプル数チェック
                if len(embeddings) > 0 and np.concatenate(embeddings, axis=0).shape[0] >= max_samples:
                    break
                    
                try:
                    if isinstance(batch_data, (tuple, list)):
                        if len(batch_data) == 2:
                            data, batch_labels = batch_data
                        else:
                            data = batch_data[0]
                            batch_labels = None
                    else:
                        data = batch_data
                        batch_labels = None
                    
                    data = data.to(device)
                    recon_data, mu, logvar = model(data)
                    
                    # 潜在表現を保存
                    embeddings.append(mu.cpu().numpy())
                    
                    # ラベルを保存（楽器ファミリー）
                    if batch_labels is not None:
                        # ラベルの型を統一
                        if isinstance(batch_labels, torch.Tensor):
                            batch_labels_list = batch_labels.cpu().numpy().tolist()
                        elif isinstance(batch_labels, np.ndarray):
                            batch_labels_list = batch_labels.tolist()
                        elif isinstance(batch_labels, list):
                            batch_labels_list = batch_labels
                        else:
                            batch_labels_list = [str(batch_labels)]
                        
                        # 各要素を文字列に変換
                        batch_labels_list = [str(label) for label in batch_labels_list]
                        labels.extend(batch_labels_list)
                    
                    # 個別の損失を計算
                    batch_recon_loss = torch.nn.functional.mse_loss(recon_data, data, reduction='none').mean(dim=1)
                    recon_losses.append(batch_recon_loss.cpu().numpy())
                    
                    if model.use_kl and logvar is not None:
                        batch_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
                        kl_divs.append(batch_kl.cpu().numpy())
                        
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # NumPy配列に変換
        if not embeddings:  # リスト段階なのでOK
            logger.warning("No data collected for metrics computation")
            return {}
            
        embeddings = np.concatenate(embeddings, axis=0)
        recon_losses = np.concatenate(recon_losses, axis=0)
        
        # kl_divs は最初リストなので len() でチェック
        if len(kl_divs) > 0:
            kl_divs = np.concatenate(kl_divs, axis=0)
        else:
            kl_divs = np.array([])  # 空の配列を定義しておく
        
        # メトリクス計算
        metrics = {}
        
        # 再構成性能
        metrics['mean_recon_loss'] = float(np.mean(recon_losses))
        metrics['std_recon_loss'] = float(np.std(recon_losses))
        
        # KL発散（配列化後は .size でチェック）
        if kl_divs.size > 0:
            metrics['mean_kl_div'] = float(np.mean(kl_divs))
            metrics['std_kl_div'] = float(np.std(kl_divs))
        
        # クラスタリング性能（ラベルが存在する場合のみ）
        if len(labels) > 0:  # labels が存在し、空でない場合のみ
            try:
                # ラベルの長さを確認
                if len(labels) == len(embeddings):
                    clustering_metrics = self.clustering_metrics.compute_clustering_metrics(embeddings, labels)
                    metrics.update(clustering_metrics)
                    
                    distance_metrics = self.clustering_metrics.compute_intra_inter_cluster_distances(embeddings, labels)
                    metrics.update(distance_metrics)
                else:
                    logger.warning(f"Label length mismatch: {len(labels)} vs {len(embeddings)}")
            except Exception as e:
                logger.warning(f"Error computing clustering metrics: {e}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
        
        # 潜在空間の統計
        metrics['latent_mean_norm'] = float(np.mean(np.linalg.norm(embeddings, axis=1)))
        metrics['latent_std_norm'] = float(np.std(np.linalg.norm(embeddings, axis=1)))
        
        logger.info(f"Computed metrics for {len(embeddings)} samples with {len(labels)} labels")
        
        return metrics