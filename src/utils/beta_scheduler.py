import numpy as np
from typing import Union

class BetaScheduler:
    """β-VAE用のKL重みスケジューラー"""
    
    def __init__(
        self,
        schedule_type: str = "linear_warmup",
        total_epochs: int = 100,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        warmup_epochs: int = 50,
        freeze_epochs: int = 10,  # 新規追加：β=0で固定するエポック数
        **kwargs
    ):
        """
        Args:
            schedule_type: "linear_warmup", "cyclical", "constant"
            total_epochs: 総エポック数
            beta_start: 開始時のβ値
            beta_end: 終了時のβ値
            warmup_epochs: ウォームアップエポック数
            freeze_epochs: β=0で固定するエポック数（通常は10）
        """
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs
        self.freeze_epochs = freeze_epochs  # 新規追加
        self.current_epoch = 0
        
        # 追加パラメータ
        self.kwargs = kwargs
        
    def step(self, epoch: int) -> float:
        """指定エポックでのβ値を計算"""
        self.current_epoch = epoch
        
        # 最初のfreeze_epochsエポックはβ=0で固定
        if epoch < self.freeze_epochs:
            return 0.0
        
        # freeze期間後のエポック数を計算
        adjusted_epoch = epoch - self.freeze_epochs
        adjusted_warmup_epochs = max(0, self.warmup_epochs - self.freeze_epochs)
        
        if self.schedule_type == "linear_warmup":
            return self._linear_warmup(adjusted_epoch, adjusted_warmup_epochs)
        elif self.schedule_type == "cyclical":
            return self._cyclical(adjusted_epoch)
        elif self.schedule_type == "constant":
            return self.beta_end
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
    
    def _linear_warmup(self, adjusted_epoch: int, adjusted_warmup_epochs: int) -> float:
        """線形ウォームアップ（freeze期間後）"""
        if adjusted_epoch < adjusted_warmup_epochs:
            # ウォームアップ期間中は線形増加
            ratio = adjusted_epoch / adjusted_warmup_epochs
            beta = self.beta_start + (self.beta_end - self.beta_start) * ratio
        else:
            # ウォームアップ後は固定
            beta = self.beta_end
        
        return beta
    
    def _cyclical(self, adjusted_epoch: int) -> float:
        """周期的スケジューリング（freeze期間後）"""
        cycle_length = self.kwargs.get('cycle_length', 50)
        
        # 現在のサイクル内での位置
        cycle_position = adjusted_epoch % cycle_length
        
        if cycle_position < cycle_length // 2:
            # サイクルの前半：β増加
            ratio = cycle_position / (cycle_length // 2)
            beta = self.beta_start + (self.beta_end - self.beta_start) * ratio
        else:
            # サイクルの後半：β減少
            ratio = (cycle_position - cycle_length // 2) / (cycle_length // 2)
            beta = self.beta_end - (self.beta_end - self.beta_start) * ratio
        
        return beta
    
    def get_beta(self) -> float:
        """現在のβ値を取得"""
        return self.step(self.current_epoch)
    
    def get_schedule_info(self) -> dict:
        """スケジューリング情報を取得"""
        return {
            'schedule_type': self.schedule_type,
            'total_epochs': self.total_epochs,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'warmup_epochs': self.warmup_epochs,
            'freeze_epochs': self.freeze_epochs,
            'current_epoch': self.current_epoch
        }