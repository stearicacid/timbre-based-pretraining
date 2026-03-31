import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns

class TradeoffAnalysis:
    def __init__(self):
        self.history = []
        
    def add_metrics(self, epoch: int, metrics: Dict[str, float]):
        """メトリクスを履歴に追加"""
        metrics['epoch'] = epoch
        self.history.append(metrics)
    
    def plot_tradeoff_curves(self, save_path: str = "tradeoff_curves.png"):
        """トレードオフ曲線を描画"""
        if not self.history:
            return
            
        # データを整理
        epochs = [h['epoch'] for h in self.history]
        recon_losses = [h.get('mean_recon_loss', 0) for h in self.history]
        kl_divs = [h.get('mean_kl_div', 0) for h in self.history]
        silhouette_scores = [h.get('silhouette_score', 0) for h in self.history]
        ari_scores = [h.get('ari', 0) for h in self.history]
        
        # 4つのサブプロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 時系列プロット
        ax1 = axes[0, 0]
        ax1.plot(epochs, recon_losses, 'b-', label='Reconstruction Loss', linewidth=2)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, kl_divs, 'r-', label='KL Divergence', linewidth=2)
        ax1_twin.plot(epochs, silhouette_scores, 'g-', label='Silhouette Score', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Reconstruction Loss', color='b')
        ax1_twin.set_ylabel('KL Div / Silhouette', color='r')
        ax1.set_title('Training Progress')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. 再構成 vs KL のトレードオフ
        ax2 = axes[0, 1]
        scatter = ax2.scatter(recon_losses, kl_divs, c=silhouette_scores, 
                            cmap='viridis', alpha=0.7, s=50)
        ax2.set_xlabel('Reconstruction Loss')
        ax2.set_ylabel('KL Divergence')
        ax2.set_title('Reconstruction vs KL (colored by Silhouette)')
        plt.colorbar(scatter, ax=ax2, label='Silhouette Score')
        
        # 3. 再構成 vs クラスタリング のトレードオフ
        ax3 = axes[1, 0]
        scatter2 = ax3.scatter(recon_losses, silhouette_scores, c=kl_divs, 
                             cmap='plasma', alpha=0.7, s=50)
        ax3.set_xlabel('Reconstruction Loss')
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title('Reconstruction vs Clustering (colored by KL)')
        plt.colorbar(scatter2, ax=ax3, label='KL Divergence')
        
        # 4. 3D トレードオフ（投影）
        ax4 = axes[1, 1]
        # パレート最適解の近似
        pareto_indices = self._find_pareto_optimal(recon_losses, kl_divs, silhouette_scores)
        
        ax4.scatter(recon_losses, silhouette_scores, alpha=0.3, s=30, label='All epochs')
        ax4.scatter([recon_losses[i] for i in pareto_indices], 
                   [silhouette_scores[i] for i in pareto_indices], 
                   c='red', s=100, alpha=0.8, label='Pareto optimal')
        ax4.set_xlabel('Reconstruction Loss')
        ax4.set_ylabel('Silhouette Score')
        ax4.set_title('Pareto Optimal Solutions')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 最適解の情報を出力
        if pareto_indices:
            print("\n=== Pareto Optimal Solutions ===")
            for i, idx in enumerate(pareto_indices):
                metrics = self.history[idx]
                print(f"Solution {i+1}:")
                print(f"  Epoch: {metrics['epoch']}")
                print(f"  Reconstruction Loss: {metrics.get('mean_recon_loss', 0):.4f}")
                print(f"  KL Divergence: {metrics.get('mean_kl_div', 0):.4f}")
                print(f"  Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
                print(f"  ARI: {metrics.get('ari', 0):.4f}")
                print()
    
    def _find_pareto_optimal(self, recon_losses: List[float], kl_divs: List[float], 
                            silhouette_scores: List[float]) -> List[int]:
        """パレート最適解を見つける（簡易版）"""
        # 再構成損失は最小化、シルエットスコアは最大化
        # KL発散は適度な値が望ましい
        
        pareto_indices = []
        n = len(recon_losses)
        
        for i in range(n):
            is_pareto = True
            for j in range(n):
                if i != j:
                    # i が j に支配されるかチェック
                    if (recon_losses[j] <= recon_losses[i] and 
                        silhouette_scores[j] >= silhouette_scores[i] and
                        (recon_losses[j] < recon_losses[i] or silhouette_scores[j] > silhouette_scores[i])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def find_optimal_hyperparameters(self) -> Dict[str, float]:
        """最適なハイパーパラメータを提案"""
        if not self.history:
            return {}
        
        # 正規化されたスコアを計算
        recon_losses = [h.get('mean_recon_loss', 0) for h in self.history]
        silhouette_scores = [h.get('silhouette_score', 0) for h in self.history]
        
        # スコアを正規化（0-1）
        min_recon, max_recon = min(recon_losses), max(recon_losses)
        min_sil, max_sil = min(silhouette_scores), max(silhouette_scores)
        
        best_score = -float('inf')
        best_metrics = None
        
        for metrics in self.history:
            norm_recon = (metrics.get('mean_recon_loss', 0) - min_recon) / (max_recon - min_recon + 1e-8)
            norm_sil = (metrics.get('silhouette_score', 0) - min_sil) / (max_sil - min_sil + 1e-8)
            
            # 総合スコア（重み付き）
            composite_score = 0.4 * (1 - norm_recon) + 0.6 * norm_sil
            
            if composite_score > best_score:
                best_score = composite_score
                best_metrics = metrics
        
        return best_metrics or {}