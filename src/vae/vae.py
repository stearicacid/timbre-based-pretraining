import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import random
import time  

from src.losses import loss_function

class HarmonicVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 45,
        latent_dim: int = 10,
        hidden_dims: List[int] = [128, 64, 32],
        beta: float = 1.0,
        loss: dict = None,
        use_kl: bool = True,
        # contractive_weight: float = 0.0,
        free_bits: float = 0.0,
        use_ddsp_decoder: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.use_kl = use_kl
        # self.contractive_weight = contractive_weight
        self.free_bits = free_bits
        self.use_ddsp_decoder = use_ddsp_decoder
        
        # self.identity_test = False
        
        if loss is not None:
            self.beta = loss.get('beta', beta)
            self.recon_weight = loss.get('recon_weight', 1.0)
            self.use_kl = loss.get('use_kl', use_kl)
            self.free_bits = loss.get('free_bits', free_bits)
            self.use_ddsp_decoder = loss.get('use_ddsp_decoder', use_ddsp_decoder)
            # self.identity_test = loss.get('identity_test', False)

            self.use_triplet = loss.get('use_triplet', False)
            self.triplet_weight = loss.get('triplet_weight', 0.01)
            self.triplet_margin = loss.get('triplet_margin', 0.3)
            # self.triplet_mining_strategy = loss.get('triplet_mining_strategy', 'batch_all')
        else:
            self.beta = beta
            self.recon_weight = 1.0
            self.use_triplet = False
            self.triplet_weight = 0.01
            self.triplet_margin = 0.3
            # self.triplet_mining_strategy = 'batch_all'

        self.current_beta = self.beta if self.use_kl else 0.0
        

        # Encoder 
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),  
            ])
            if i < len(hidden_dims) - 1:
                encoder_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        if self.use_kl:
            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

            nn.init.xavier_uniform_(self.fc_mu.weight)
            nn.init.zeros_(self.fc_mu.bias)  
            
            nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.1)
            nn.init.constant_(self.fc_logvar.bias, -2.0)
        else:
            self.fc_latent = nn.Linear(hidden_dims[-1], latent_dim)
            nn.init.xavier_uniform_(self.fc_latent.weight)
            nn.init.zeros_(self.fc_latent.bias)
        
        # Decoder 
        decoder_layers = []
        prev_dim = latent_dim
        
        reversed_dims = list(reversed(hidden_dims))
        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU()
            ])
            if i < len(reversed_dims) - 1:
                decoder_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()

        
        print(f"Final encoder: {self.encoder}")
        print(f"Final decoder: {self.decoder}")

    def _initialize_weights(self):
        """全ての線形層をXavier初期化"""
            
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.encoder(x)
        if self.use_kl:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        else:
            latent = self.fc_latent(h)
            return latent, None
    
    def reparameterize(self, mu: torch.Tensor, logvar: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_kl and logvar is not None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        
        if self.use_ddsp_decoder:
            from src.utils.normalization import ddsp_style_normalization
            return ddsp_style_normalization(logits)
        else:
            return logits
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def set_beta(self, beta: float):
        if self.use_kl:
            self.current_beta = beta
        else:
            self.current_beta = 0.0
    

    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._batch_all_vectorized(embeddings, labels)

    def _batch_all_vectorized(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, embeddings, p=2)
        N = dist.size(0)

        ap = dist.unsqueeze(2)               # [N,N,1]
        an = dist.unsqueeze(1)               # [N,1,N]
        losses = ap - an + self.triplet_margin  # [N,N,N]

        # Create masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N,N]

        # verify that i != j (anchor != positive)
        idxs = torch.arange(N, device=labels.device)
        neq = idxs.unsqueeze(0) != idxs.unsqueeze(1)            # i != j

        # positive_mask: a≠p && same label
        pos_mask = labels_eq & neq                              

        # negative_mask: a≠n && different label
        neg_mask = ~labels_eq                                   

        # mask[i, j, k] は (labels[i] == labels[j] and i != j) and (labels[i] != labels[k]) が True の場合のみ True
        mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)    # [N,N,N]

        losses_for_valid_triplets = losses[mask]
        final_losses = torch.clamp(losses_for_valid_triplets, min=0.0)

        if final_losses.numel() > 0:
            return final_losses.mean()

        else:
            return torch.tensor(0.0, device=embeddings.device)

