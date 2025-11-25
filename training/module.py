import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from typing import List

class InstanceEncoder(nn.Module):
    def __init__(self,
                 num_genes: int,
                 d_model: int = 256,
                 n_heads: int = 1,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_genes = num_genes

        self.gene_embed = nn.Embedding(num_genes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        """
        input_ids: (B, L)
        attn_mask: (B, L), True for real tokens, False for padding
        """
        B, L = input_ids.shape
        tok_emb = self.gene_embed(input_ids)  # (B, L, d_model)

        # Transformer expects True for PAD, so invert
        key_padding_mask = ~attn_mask  # (B, L)

        x = self.encoder(tok_emb, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        # mean-pool over non-padding tokens
        attn_mask_f = attn_mask.unsqueeze(-1).float()   # (B, L, 1)
        x_masked = x * attn_mask_f
        sum_emb = x_masked.sum(dim=1)                   # (B, d_model)
        lengths = attn_mask_f.sum(dim=1).clamp(min=1e-6)
        cell_emb = sum_emb / lengths                    # (B, d_model)
        return cell_emb


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: List[int] = (1024, 1024),
        dropout: float = 0.,
        residual: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = output_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                if batchnorm:
                    self.network.append(
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim[i]),
                            ## either normalize batchwise when (batch, cell, gene)
                            ## will first become (batch x cell, gene) then normalize gene wise
                            ## or normalize gene wise (cell, gene)
                            nn.BatchNorm1d(hidden_dim[i]),
                            nn.PReLU(),
                            # nn.Mish(),
                        )
                    )
                else:
                    self.network.append(
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim[i]),
                            # nn.BatchNorm1d(hidden_dim[i]),
                            nn.PReLU(),
                            # nn.Mish(),
                        )
                    )
            else:  # hidden layers
                if batchnorm:
                    self.network.append(
                        nn.Sequential(
                            nn.Dropout(p=dropout),
                            nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                            nn.BatchNorm1d(hidden_dim[i]),
                            nn.PReLU(),
                            # nn.Mish(),
                        )
                    )
                else:
                    self.network.append(
                        nn.Sequential(
                            nn.Dropout(p=dropout),
                            nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                            # nn.BatchNorm1d(hidden_dim[i]),
                            nn.PReLU(),
                            # nn.Mish(),
                        )
                    )
        # output layer
        if hidden_dim != []:
            self.network.append(nn.Linear(hidden_dim[-1], self.latent_dim))
        else:
            self.network.append(nn.Linear(input_dim, self.latent_dim))

    def forward(self, x) -> torch.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x
