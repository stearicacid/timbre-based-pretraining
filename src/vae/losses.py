import torch
import torch.nn.functional as F
from typing import Optional
import time  
from typing import Callable

def compute_loss(recon_x: torch.Tensor, x: torch.Tensor, 
                    mu: torch.Tensor, logvar: Optional[torch.Tensor] = None,
                    labels: Optional[torch.Tensor] = None,
                    *,
                    use_ddsp_decoder: bool = False,
                    free_bits: float = 0.0,
                    recon_weight: float = 1.0,
                    beta: float = 1.0,
                    triplet_weight: float = 0.01,
                    mine_triplets_fn: Optional[Callable] = None,
                    loss_weights: Optional[dict] = None,
                ) -> dict:
    
    loss_weights = loss_weights or {'mse': 1.0}
    
    x_processed = x
    if use_ddsp_decoder:
        if torch.max(x) > 1.1 or torch.min(x) < -0.1: 
            from src.utils.normalization import ddsp_style_normalization
            x_processed = ddsp_style_normalization(x)
    
    mse_loss = F.mse_loss(recon_x, x_processed, reduction='mean')
    recon_loss = mse_loss

    if logvar is None:
        raise ValueError("logvar must not be None in VAE mode")

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
    kl_div = kl_per_dim.sum(dim=1).mean()


    triplet_loss = torch.tensor(0.0, device=x.device)
    triplet_time = 0.0
    if labels is not None and len(labels.unique()) > 1:
        labels_flat = labels.flatten() if labels.dim() > 1 else labels
        start_time = time.time()
        triplet_loss = mine_triplets_fn(mu, labels_flat)
        triplet_time = time.time() - start_time
    
    total_loss = (recon_weight * recon_loss + 
                    beta * kl_div +
                    triplet_weight * triplet_loss)
    
    actual_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    
    mode_str = 'VAE-DDSP+Triplet' if use_ddsp_decoder else 'VAE+Triplet'
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'mse_loss': mse_loss,
        'kl_div': kl_div,  
        'actual_kl': actual_kl,  
        'triplet_loss': triplet_loss,
        'beta': beta,
        'free_bits': free_bits,
        'mode': mode_str,
        'triplet_time': triplet_time
    }