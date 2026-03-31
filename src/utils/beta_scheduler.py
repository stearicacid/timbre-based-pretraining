import numpy as np
from typing import Union

class BetaScheduler:
    """Scheduler for β-VAE"""
    
    def __init__(
        self,
        schedule_type: str = "linear_warmup",
        total_epochs: int = 100,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        warmup_epochs: int = 50,
        freeze_epochs: int = 10,  
        **kwargs
    ):
        """
        Args:
            schedule_type: "linear_warmup", "cyclical", "constant"
            total_epochs: number of epochs for training 
            beta_start: starting β value
            beta_end: ending β value
            warmup_epochs: number of warmup epochs
            freeze_epochs: number of epochs to freeze β at 0
        """
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs
        self.freeze_epochs = freeze_epochs  
        self.current_epoch = 0
        self.kwargs = kwargs
        
    def step(self, epoch: int) -> float:
        """Calculate the value of β for the given epoch."""
        self.current_epoch = epoch
        if epoch < self.freeze_epochs:
            return 0.0
        
        adjusted_epoch = epoch - self.freeze_epochs
        adjusted_warmup_epochs = max(0, self.warmup_epochs - self.freeze_epochs)
        
        if self.schedule_type == "linear_warmup":
            return self._linear_warmup(adjusted_epoch, adjusted_warmup_epochs)
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
    
    def _linear_warmup(self, adjusted_epoch: int, adjusted_warmup_epochs: int) -> float:
        if adjusted_epoch < adjusted_warmup_epochs and adjusted_warmup_epochs > 0:
            ratio = adjusted_epoch / adjusted_warmup_epochs
            beta = self.beta_start + (self.beta_end - self.beta_start) * ratio
        else:
            beta = self.beta_end     
        return beta
    
    def get_beta(self) -> float:
        return self.step(self.current_epoch)
    
    def get_schedule_info(self) -> dict:
        return {
            'schedule_type': self.schedule_type,
            'total_epochs': self.total_epochs,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'warmup_epochs': self.warmup_epochs,
            'freeze_epochs': self.freeze_epochs,
            'current_epoch': self.current_epoch
        }