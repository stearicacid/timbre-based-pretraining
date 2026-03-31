import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import random
import time  

def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                    mu: torch.Tensor, logvar: Optional[torch.Tensor] = None,
                    labels: Optional[torch.Tensor] = None,
                    loss_weights: dict = None,
                    logger=None) -> dict:  
    
    loss_weights = loss_weights or {'mse': 1.0}
    
    x_processed = x
    if self.use_ddsp_decoder:
        if torch.max(x) > 1.1 or torch.min(x) < -0.1: 
            from src.utils.normalization import ddsp_style_normalization
            x_processed = ddsp_style_normalization(x)
    
    mse_loss = F.mse_loss(recon_x, x_processed, reduction='mean')
    recon_loss = mse_loss
    
    free_bits = getattr(self, 'free_bits', 0.0)
    
    if self.use_kl and logvar is not None:
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
        kl_div = kl_per_dim.sum(dim=1).mean()
    else:
        kl_div = torch.tensor(0.0, device=x.device)

    triplet_loss = torch.tensor(0.0, device=x.device)
    triplet_time = 0.0
    if self.use_triplet and labels is not None and len(labels.unique()) > 1:
        try:
            labels_flat = labels.flatten() if labels.dim() > 1 else labels
            start_time = time.time()
            triplet_loss = self.mine_triplets(mu, labels_flat)
            triplet_time = time.time() - start_time
            # if logger is not None:
            #     logger.info(f"[Triplet profiling] mine_triplets: {triplet_time:.4f} sec")
        except Exception as e:
            triplet_loss = torch.tensor(0.0, device=x.device)
    
    total_loss = (self.recon_weight * recon_loss + 
                    self.current_beta * kl_div +
                    self.triplet_weight * triplet_loss)
    
    actual_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean() if self.use_kl and logvar is not None else torch.tensor(0.0, device=x.device)
    
    mode_str = 'VAE-DDSP' if self.use_ddsp_decoder else ('VAE' if self.use_kl else 'AE')
    if self.use_triplet:
        mode_str += '+Triplet'
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'mse_loss': mse_loss,
        'kl_div': kl_div,  
        'actual_kl': actual_kl,  
        'triplet_loss': triplet_loss,
        'beta': self.current_beta,
        'free_bits': free_bits,
        'mode': mode_str,
        'triplet_time': triplet_time
    }